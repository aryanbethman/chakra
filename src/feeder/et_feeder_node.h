#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "et_def.pb.h"

namespace Chakra {

class ETFeederNode {
 public:
  ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node);
  std::shared_ptr<ChakraProtoMsg::Node> getChakraNode();
  void addChild(std::shared_ptr<ETFeederNode> node);
  std::vector<std::shared_ptr<ETFeederNode>> getChildren();
  void addDepUnresolvedParentID(uint64_t node_id);
  std::vector<uint64_t> getDepUnresolvedParentIDs();
  void setDepUnresolvedParentIDs(
      std::vector<uint64_t> const& dep_unresolved_parent_ids);
  // Consume one not-yet-consumed occurrence of `parent_id` among the node's
  // data dependencies. Returns whether one was found.
  bool consumeDataDep(uint64_t parent_id);
  // Data dependencies not yet consumed.
  int remainingDataDeps() const;

  const ChakraProtoMsg::AttributeProto& get_other_attr(
      const std::string& attr_name) const;
  bool has_other_attr(const std::string& attr_name) const;

  uint64_t id();
  std::string name();
  bool is_cpu_op();
  ChakraProtoMsg::NodeType type();
  uint64_t runtime();
  uint64_t num_ops();
  uint32_t tensor_loc();
  uint32_t tensor_device();
  uint32_t tensor_channel();
  uint64_t tensor_size();
  ChakraProtoMsg::CollectiveCommType comm_type();
  uint32_t comm_priority();
  uint64_t comm_size();
  uint32_t comm_src();
  uint32_t comm_dst();
  uint32_t comm_tag();
  std::string pg_name();
  std::string get_inputs_values() const;
  std::string get_inputs_shapes() const;
  std::string get_inputs_types() const;
  std::string get_outputs_values() const;
  std::string get_outputs_shapes() const;
  std::string get_outputs_types() const;
  void printNode();

 private:
  void assign_attr_val(
      std::shared_ptr<ChakraProtoMsg::Node> node,
      int i,
      void* member);

  std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
  std::unordered_set<std::shared_ptr<ETFeederNode>> children_set_{};
  std::vector<std::shared_ptr<ETFeederNode>> children_vec_{};
  std::vector<uint64_t> dep_unresolved_parent_ids_{};
  // Dependency bookkeeping lives here rather than in node_, so node_ can be
  // an immutable execution-template node shared by every rank and batch
  // instead of a per-feeder copy. Bit i of consumed_deps_ marks
  // node_->data_deps(i) consumed; nodes with more than 64 dependencies spill
  // to consumed_deps_overflow_.
  int remaining_data_deps_ = 0;
  uint64_t consumed_deps_ = 0;
  std::vector<bool> consumed_deps_overflow_{};
  std::unordered_map<std::string, const ChakraProtoMsg::AttributeProto&>
      other_attrs_{};

  uint64_t id_;
  std::string name_;
  bool is_cpu_op_;
  uint64_t runtime_;
  uint64_t num_ops_;
  uint32_t tensor_loc_;
  uint32_t tensor_device_;
  uint32_t tensor_channel_;
  uint64_t tensor_size_;
  ChakraProtoMsg::CollectiveCommType comm_type_;
  uint32_t comm_priority_;
  uint64_t comm_size_;
  uint32_t comm_src_;
  uint32_t comm_dst_;
  uint32_t comm_tag_;
  std::string pg_name_;
  std::string inputs_values_;
  std::string inputs_shapes_;
  std::string inputs_types_;
  std::string outputs_values_;
  std::string outputs_shapes_;
  std::string outputs_types_;
};

} // namespace Chakra
