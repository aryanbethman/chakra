#include "et_feeder.h"

#include <chrono>
#include <iostream>

using namespace std;
using namespace Chakra;

ETFeeder::ETFeeder(string filename)
    : trace_(make_unique<ProtoInputStream>(filename)), window_size_(4096 * 256), et_complete_(false) {
  initialiseTrace();
}

ETFeeder::ETFeeder(shared_ptr<const string> payload)
    : trace_(make_unique<ProtoInputStream>(std::move(payload))), window_size_(4096 * 256), et_complete_(false) {
  initialiseTrace();
}

ETFeeder::ETFeeder(shared_ptr<const AstraSim::RankEtTemplate> rank_template)
    : rank_template_(std::move(rank_template)), window_size_(4096 * 256), et_complete_(false) {
  if (rank_template_ == nullptr || rank_template_->nodes == nullptr) {
    throw invalid_argument("Missing direct execution template");
  }
  const auto started = std::chrono::steady_clock::now();
  readNextWindow();
  AstraSim::record_direct_template_feeder_init(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count());
}

void ETFeeder::initialiseTrace() {
  if (trace_ == nullptr || !trace_->is_open()) {
    throw runtime_error("Failed to open execution trace");
  }

  try {
    readGlobalMetadata();
    readNextWindow();
  } catch (const std::exception& e) {
    cerr << "Error in constructor: " << e.what() << endl;
    throw;
  }
}

ETFeeder::~ETFeeder() {}

void ETFeeder::addNode(shared_ptr<ETFeederNode> node) {
  dep_graph_[node->getChakraNode()->id()] = node;
}

void ETFeeder::removeNode(uint64_t node_id) {
  dep_graph_.erase(node_id);

  if (!et_complete_ && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
}

bool ETFeeder::hasNodesToIssue() {
  return !(dep_graph_.empty() && dep_free_node_queue_.empty());
}

shared_ptr<ETFeederNode> ETFeeder::getNextIssuableNode() {
  if (dep_free_node_queue_.size() != 0) {
    shared_ptr<ETFeederNode> node = dep_free_node_queue_.top();
    dep_free_node_id_set_.erase(node->getChakraNode()->id());
    dep_free_node_queue_.pop();
    return node;
  } else {
    return nullptr;
  }
}

void ETFeeder::pushBackIssuableNode(uint64_t node_id) {
  shared_ptr<ETFeederNode> node = dep_graph_[node_id];
  dep_free_node_id_set_.emplace(node_id);
  dep_free_node_queue_.emplace(node);
}

shared_ptr<ETFeederNode> ETFeeder::lookupNode(uint64_t node_id) {
  try {
    return dep_graph_.at(node_id);
  } catch (const std::out_of_range& e) {
    std::cerr << "looking for node_id=" << node_id
              << " in dep graph, however, not loaded yet" << std::endl;
    throw(e);
  }
}

void ETFeeder::freeChildrenNodes(uint64_t node_id) {
  shared_ptr<ETFeederNode> node = dep_graph_[node_id];
  for (auto child : node->getChildren()) {
    auto child_chakra = child->getChakraNode();
    for (auto it = child_chakra->mutable_data_deps()->begin();
         it != child_chakra->mutable_data_deps()->end();
         ++it) {
      if (*it == node_id) {
        child_chakra->mutable_data_deps()->erase(it);
        break;
      }
    }
    if (child_chakra->data_deps().size() == 0) {
      dep_free_node_id_set_.emplace(child_chakra->id());
      dep_free_node_queue_.emplace(child);
    }
  }
}

void ETFeeder::readGlobalMetadata() {
  if (trace_ == nullptr || !trace_->is_open()) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading global metadata.");
  }
  shared_ptr<ChakraProtoMsg::GlobalMetadata> pkt_msg =
      make_shared<ChakraProtoMsg::GlobalMetadata>();
  trace_->read(*pkt_msg);
}

shared_ptr<ETFeederNode> ETFeeder::readNode() {
  shared_ptr<ChakraProtoMsg::Node> pkt_msg =
      make_shared<ChakraProtoMsg::Node>();
  if (rank_template_ != nullptr) {
    if (template_node_index_ >= rank_template_->nodes->size()) {
      return nullptr;
    }
    pkt_msg->CopyFrom(*rank_template_->nodes->at(template_node_index_));
    const auto overlay = rank_template_->overlays.find(template_node_index_++);
    if (overlay != rank_template_->overlays.end()) {
      if (overlay->second.has_name) {
        pkt_msg->set_name(overlay->second.name);
      }
      if (!overlay->second.attributes.empty()) {
        unordered_map<int, ChakraProtoMsg::AttributeProto> replacements;
        for (const auto& replacement : overlay->second.attributes) {
          if (!replacements.emplace(replacement.first, replacement.second).second) {
            throw invalid_argument("duplicate template node attribute position");
          }
        }
        vector<ChakraProtoMsg::AttributeProto> retained;
        retained.reserve(pkt_msg->attr_size());
        for (const auto& attribute : pkt_msg->attr()) {
          retained.push_back(attribute);
        }
        const int total = static_cast<int>(retained.size() + replacements.size());
        pkt_msg->clear_attr();
        size_t retained_index = 0;
        for (int position = 0; position < total; ++position) {
          const auto replacement = replacements.find(position);
          if (replacement != replacements.end()) {
            pkt_msg->add_attr()->CopyFrom(replacement->second);
          } else if (retained_index < retained.size()) {
            pkt_msg->add_attr()->CopyFrom(retained.at(retained_index++));
          } else {
            throw invalid_argument("invalid template node attribute position");
          }
        }
      }
    }
  } else if (!trace_->read(*pkt_msg)) {
    return nullptr;
  }
  shared_ptr<ETFeederNode> node = make_shared<ETFeederNode>(pkt_msg);

  bool dep_unresolved = false;
  for (int i = 0; i < pkt_msg->data_deps_size(); ++i) {
    auto parent_node = dep_graph_.find(pkt_msg->data_deps(i));
    if (parent_node != dep_graph_.end()) {
      parent_node->second->addChild(node);
    } else {
      dep_unresolved = true;
      node->addDepUnresolvedParentID(pkt_msg->data_deps(i));
    }
  }

  if (dep_unresolved) {
    dep_unresolved_node_set_.emplace(node);
  }

  return node;
}

void ETFeeder::resolveDep() {
  for (auto it = dep_unresolved_node_set_.begin();
       it != dep_unresolved_node_set_.end();) {
    shared_ptr<ETFeederNode> node = *it;
    vector<uint64_t> dep_unresolved_parent_ids =
        node->getDepUnresolvedParentIDs();
    for (auto inner_it = dep_unresolved_parent_ids.begin();
         inner_it != dep_unresolved_parent_ids.end();) {
      auto parent_node = dep_graph_.find(*inner_it);
      if (parent_node != dep_graph_.end()) {
        parent_node->second->addChild(node);
        inner_it = dep_unresolved_parent_ids.erase(inner_it);
      } else {
        ++inner_it;
      }
    }
    if (dep_unresolved_parent_ids.size() == 0) {
      it = dep_unresolved_node_set_.erase(it);
    } else {
      node->setDepUnresolvedParentIDs(dep_unresolved_parent_ids);
      ++it;
    }
  }
}

void ETFeeder::readNextWindow() {
  if (rank_template_ == nullptr && (trace_ == nullptr || !trace_->is_open())) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading next window.");
  }
  uint32_t num_read = 0;
  do {
    shared_ptr<ETFeederNode> new_node = readNode();
    if (new_node == nullptr) {
      et_complete_ = true;
      break;
    }

    addNode(new_node);
    ++num_read;

    resolveDep();
  } while ((num_read < window_size_) || (dep_unresolved_node_set_.size() != 0));

  for (auto node_id_node : dep_graph_) {
    uint64_t node_id = node_id_node.first;
    shared_ptr<ETFeederNode> node = node_id_node.second;
    if ((dep_free_node_id_set_.count(node_id) == 0) &&
        (node->getChakraNode()->data_deps().size() == 0)) {
      dep_free_node_id_set_.emplace(node_id);
      dep_free_node_queue_.emplace(node);
    }
  }
}

void ETFeeder::printGraph() {
  for (auto const &pair: dep_graph_) {
      cout << "{" << pair.first << ": ";
      pair.second->printNode();
  }
}
