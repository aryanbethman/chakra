import logging
from contextlib import nullcontext
from io import BytesIO, StringIO, TextIOWrapper
from typing import Any, Dict, List, Optional
import hashlib
from enum import Enum

from ...schema.protobuf.et_def_pb2 import *
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ..third_party.utils.protolib import encodeMessage as encode_message


# Memory type is deprecated for latest version of chakra & astra-sim
# It only uses tensor_size for the remote memory issue in Workload.cc
# Added memory types using enum and add tensor_loc in node attribute in et_feeder_node.cpp
class MemoryType(Enum):
    INVALID_MEMORY = 0
    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    CXL_MEMORY = 3
    STORAGE_MEMORY = 4


class Layer:
    def __init__(self, line: str):
        try:
            col = line.strip().split()
            if col[0] == 'EXPERT': # If Expert Flag
                self.name = col[0]
                self.expert_num = col[1]
                self.is_expert = True
                self.is_pim = False
                self.comm_node = None
                self.comp_node = None
                self.comm_type = "ALLTOALL"
            elif col[0] == 'PIM': # If PIM Flag
                self.name = col[0]
                self.pim_num = col[1]
                self.is_expert = False
                self.is_pim = True
                self.comm_node = None
                self.comp_node = None
                self.comm_type = "NONE"
            else:
                self.is_expert = False
                self.is_pim = False
                self.name = col[0]

                # compuation
                self.comp_time = int(col[1])
                self.comp_node = None

                # memory
                self.input_memory_loc = str(col[2])
                self.input_memory_size = int(col[3])
                self.input_memory_node = None
                self.weight_memory_loc = str(col[4])
                self.weight_memory_size = int(col[5])
                self.weight_memory_node = None
                self.output_memory_loc = str(col[6])
                self.output_memory_size = int(col[7])
                self.output_memory_node = None

                # communication
                self.comm_type = str(col[8])
                self.comm_size = int(col[9])
                self.comm_node = None

                self.misc = str(col[10])
        except:
            raise ValueError(f"Cannot parse the following layer -- \"{line}\"")
        
class LLMConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_npus: int,
        npu_offset: int = 0,
        local_offloading: bool = False,
        input_text: Optional[str] = None,
    ):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.npu_offset = npu_offset
        self.local_offloading = local_offloading
        self.input_text = input_text
        self.next_node_id = 0

        # For send & recv nodes
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()
        self._payload_streams = None
        self._template_collector = None
        self._compact_metadata = False
        self._global_metadata = None

    def _reset_conversion_state(self) -> None:
        self.next_node_id = 0
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()
        self._global_metadata = None

    def _open_input(self):
        if self.input_text is not None:
            return StringIO(self.input_text)
        return open(self.input_filename, "r")

    def _open_output_stream(self, npu_id: int):
        """Return a binary ET sink for a rank.

        The legacy converter writes a rank-specific file. The in-memory API
        swaps only this sink, so both paths use the same encoding logic.
        """
        if self._template_collector is not None:
            return self._template_collector.open_rank(npu_id)
        if self._payload_streams is None:
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            return open(output_filename, "wb")

        stream = BytesIO()
        self._payload_streams[npu_id] = stream
        return nullcontext(stream)

    @staticmethod
    def _emit_message(stream, message) -> None:
        """Write a framed ET message or capture its template representation."""
        if hasattr(stream, "write_message"):
            stream.write_message(message)
        else:
            encode_message(stream, message)

    def get_global_metadata(self):
        if self._global_metadata is None:
            if self._compact_metadata:
                trace_identity = hashlib.sha256(
                    (self.input_text or "").encode("utf-8")
                ).hexdigest()
                attr = [
                    ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
                    ChakraAttr(name="trace_sha256", string_val=trace_identity),
                ]
            else:
                input_text = self.input_text
                if input_text is None:
                    with self._open_input() as input_file:
                        input_text = input_file.read()
                attr = [
                    ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
                    ChakraAttr(name="input_file", string_val=input_text),
                ]
            self._global_metadata = GlobalMetadata(attr=attr)
        return self._global_metadata
    
    def get_layers(self, f: TextIOWrapper) -> List[Layer]:
        layers: List[Layer] = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def get_next_node_id(self) -> int:
        ret = self.next_node_id
        self.next_node_id += 1
        return ret
    
    def get_next_comm_tag(self) -> int:
        ret = self.next_comm_tag
        self.next_comm_tag += 1
        return ret

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.get_next_node_id()
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, layer_name: str, comp_time: int) -> Any:
        node = self.get_node("COMP_NODE_" + layer_name, COMP_NODE)
        node.duration_micros = comp_time
        return node
    
    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0
    
    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{layer_name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        return node

    def get_comm_node(self, is_send: bool, layer_name: str, comm_type: str, comm_size: int,
                           comm_src: int, comm_dst: int, id: int = 0) -> Any:
        if is_send:
            node = self.get_node(
                    f"COMM_SEND_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_SEND_NODE)
        else:
            node = self.get_node(
                    f"COMM_RECV_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=comm_src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=comm_dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        comm_key = f"{comm_src}_{comm_dst}_{id}"
        if comm_key in self.comm_tag_dict:
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.comm_tag_dict[comm_key]))
        else:
            new_tag = self.get_next_comm_tag()
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=new_tag))
            self.comm_tag_dict[comm_key] = new_tag

        # check if SEND/RECV pair have same tags
        # print(f"name: {node.name}, src: {node.comm_src}, dst: {node.comm_dst}, size: {node.comm_size}, key: {comm_key}, tag: {node.comm_tag}")
        return node
    
    def get_mem_type(self, mem_type: str) -> int:
        mem_type = mem_type.split(':')[0]  # Exclude the device number if present
        if mem_type == "LOCAL":
            return MemoryType.LOCAL_MEMORY.value
        elif mem_type == "REMOTE":
            return MemoryType.REMOTE_MEMORY.value
        elif mem_type == "CXL":
            return MemoryType.CXL_MEMORY.value
        elif mem_type == "STORAGE":
            return MemoryType.STORAGE_MEMORY.value
        return MemoryType.INVALID_MEMORY.value
    
    def get_mem_device(self, mem_type: str) -> int:
        """
        Extract the device index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 1
        - "REMOTE:1.3"      -> returns 1  (device = 1, channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            dev_part = parts[1].split(".", 1)[0]
            if dev_part.isdigit():
                return int(dev_part)
        return 0  # Default device number


    def get_mem_channel(self, mem_type: str) -> int:
        """
        Extract the channel index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 0  (no channel specified)
        - "REMOTE:1.3"      -> returns 3  (channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            sub = parts[1].split(".", 1)
            if len(sub) == 2:
                chan_part = sub[1]
                if chan_part.isdigit():
                    return int(chan_part)
        return 0  # Default channel number


    def get_memory_load_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_LOAD_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_memory_store_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_STORE_NODE_" + layer_name + "_" + tensor_type, MEM_STORE_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_pim_compute_node(self, layer_name: str, tensor_type: str, comp_time: int, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("PIM_COMP_NODE_" + layer_name + "_" + tensor_type, PIM_COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_channel", uint32_val=self.get_mem_channel(mem_type)))
        return node
        
    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def convert_common(self, f: TextIOWrapper, num_layers: int, num_npu_group: int, pipeline_boundaries=None):
        layers: list[Layer] = self.get_layers(f)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        if pipeline_boundaries is not None:
            # Boundaries are exclusive indices into the original operator list.
            # Account for optional leading KV load/evict records removed above.
            pipeline_boundaries = [boundary - ev_ld_cnt for boundary in pipeline_boundaries]
            if (len(pipeline_boundaries) != num_npu_group or
                    pipeline_boundaries[-1] != num_layers or
                    any(a >= b for a, b in zip([0] + pipeline_boundaries, pipeline_boundaries))):
                raise ValueError("invalid pipeline block boundaries")
        else:
            layers_per_group = num_layers // num_npu_group
            remain_layers = num_layers % num_npu_group

        layer_start = 0
        layer_end = 0

        for npu_group in range(num_npu_group):
            layer_start = layer_end
            if pipeline_boundaries is not None:
                layer_end = pipeline_boundaries[npu_group]
            else:
                layer_end = layer_start + layers_per_group + (1 if remain_layers > 0 else 0)
                if layer_end >= num_layers:
                    layer_end = num_layers
            for npu_offset in range(npus_per_group):
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                first_comp_node = True
                with self._open_output_stream(npu_id) as g:
                    global_metadata = self.get_global_metadata()
                    self._emit_message(g, global_metadata)
                    if evict != None:
                        self._emit_message(g, evict)
                    if load != None:
                        self._emit_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        self._emit_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert or layers[layer_start].is_pim:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            self._emit_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            self._emit_message(g, receive_input_node)

                    expert_start = False
                    pim_start = False
                    attn_remain = False # to handle remaining prefill attention after pim
                    pim_parent_nodes = []
                    pim_comp_nodes = []
                    past_pim_comp_nodes = []
                    last_batch_type = "BATCH_1"
                    layer_num = layer_start
                    while expert_start or pim_start or attn_remain or layer_num < layer_end:
                        if not layers[layer_num].is_expert and not layers[layer_num].is_pim: 
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                self._emit_message(g, weight_load_node)
                            # Compute
                            if layers[layer_num].comp_time != 0 and not pim_start: # pim computation is handled pim_comp_node
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name, 
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node

                                # handle pim parent nodes, and if prefill attention remains wait until all attention is done (before o_proj)
                                if len(pim_parent_nodes) != 0:
                                    if attn_remain:
                                        for parent in pim_parent_nodes: 
                                            self.add_parent(comp_node, parent)
                                    pim_parent_nodes = [] # reset pim parent nodes

                                    if "attn" in layers[layer_num].name:
                                        attn_remain = False
                                else: 
                                    if first_comp_node:
                                        if npu_group == 0:
                                            self.add_parent(comp_node, input_load_node)
                                        else:
                                            self.add_parent(comp_node, receive_input_node)
                                        if evict != None:
                                            self.add_parent(comp_node, evict)
                                        if load != None:
                                            self.add_parent(comp_node, load)
                                        if layers[layer_num].weight_memory_node != None:
                                            self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                        first_comp_node = False
                                    else:
                                        if layers[layer_num].weight_memory_node != None:
                                            self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                        if layers[layer_num - 1].comm_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                        elif layers[layer_num - 1].comp_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                        else:
                                            print(layers[layer_num - 2].name, layers[layer_num - 1].name, layers[layer_num].name)
                                            self.add_parent(comp_node, layers[layer_num - 2].comp_node)

                                # handle pim_compute_mode dependency & should not be remaining attention
                                if not attn_remain and len(pim_comp_nodes) != 0:
                                    if layers[layer_num].misc == "NONE": # no sub-batch interleaving
                                        for pim_comp in pim_comp_nodes:
                                            self.add_parent(comp_node, pim_comp)
                                        pim_comp_nodes = [] # reset pim comp nodes
                                    elif layers[layer_num].misc != last_batch_type:
                                        if len(past_pim_comp_nodes) != 0:
                                            for pim_comp in past_pim_comp_nodes:
                                                self.add_parent(comp_node, pim_comp)
                                        past_pim_comp_nodes = pim_comp_nodes # update past pim comp nodes
                                        pim_comp_nodes = [] # reset pim comp nodes
                                        last_batch_type = layers[layer_num].misc

                                self._emit_message(g, comp_node)

                            # PIM compute
                            if pim_start:
                                pim_comp_node = self.get_pim_compute_node(
                                    layers[layer_num].name,
                                    "PIM",
                                    layers[layer_num].comp_time,
                                    layers[layer_num].input_memory_loc,
                                    layers[layer_num].input_memory_size + layers[layer_num].output_memory_size # load/store from pim
                                )
                                pim_comp_nodes.append(pim_comp_node)
                                for parent in pim_parent_nodes:
                                    self.add_parent(pim_comp_node, parent)
                                self._emit_message(g, pim_comp_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                self._emit_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            if expert_start == False and use_comm:
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num-1].output_memory_size)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                self._emit_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                                # End of expert, add ALLTOALL communication after expert computation
                                if use_comm:
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num+1].input_memory_size)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    self._emit_message(g, comm_coll_node)
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                        # pim layer starts
                        elif layers[layer_num].is_pim:
                            pim_start = True
                            # check attention end
                            if layers[layer_num].pim_num == 'END':
                                pim_start = False
                                if "attn" in layers[layer_num + 1].name:
                                    attn_remain = True # prefill attn remains
                                layer_num += 1
                                continue
                            # add pim parent nodes for dependency
                            elif int(layers[layer_num].pim_num) == 0:
                                if first_comp_node:
                                    if npu_group == 0:
                                        pim_parent_nodes.append()
                                    else:
                                        pim_parent_nodes.append(receive_input_node)
                                    if evict != None:
                                        pim_parent_nodes.append(evict)
                                    if load != None:
                                        pim_parent_nodes.append(load)
                                    # discarded weight parent because attention has no weight
                                    first_comp_node = False
                                else:
                                    # discarded weight parent because attention has no weight
                                    if layers[layer_num - 1].comm_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                                    else:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                            # round robin assignment
                            pim_id = int(layers[layer_num].pim_num) % npus_per_group
                            if npu_offset != pim_id:
                                # go to next attention
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_pim:
                                        break
                            else:
                                layer_num += 1

                    # update new layer_end
                    layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Store output (for the last layer)
                        output_store_node = self.get_memory_store_node(
                            layers[layer_end - 1].name,
                            "OUTPUT",
                            layers[layer_end - 1].output_memory_loc,
                            layers[layer_end - 1].input_memory_size,
                        )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(output_store_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(output_store_node, comp_node)
                        else:
                            self.add_parent(output_store_node, layers[layer_end - 2].comp_node)
                        self._emit_message(g, output_store_node)
                    else:
                        if layers[layer_end - 1].is_expert or layers[layer_end - 1].is_pim:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        self._emit_message(g, send_output_node)
            if pipeline_boundaries is None:
                remain_layers -= 1
    
    def convert_prefill(self, f: TextIOWrapper, num_layers: int, num_npu_group: int, pipeline_boundaries=None):
        layers: list[Layer] = self.get_layers(f)
        # There will be no pim operation in prefill (PIM cannot perform GEMM)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        if pipeline_boundaries is not None:
            if (len(pipeline_boundaries) != num_npu_group or
                    pipeline_boundaries[-1] != num_layers or
                    any(a >= b for a, b in zip([0] + pipeline_boundaries, pipeline_boundaries))):
                raise ValueError("invalid pipeline block boundaries")
        else:
            layers_per_group = num_layers // num_npu_group
            remain_layers = num_layers % num_npu_group

        layer_start = 0
        layer_end = 0

        for npu_group in range(num_npu_group):
            layer_start = layer_end
            if pipeline_boundaries is not None:
                layer_end = pipeline_boundaries[npu_group]
            else:
                layer_end = layer_start + layers_per_group + (1 if remain_layers > 0 else 0)
                if layer_end >= num_layers:
                    layer_end = num_layers
            for npu_offset in range(npus_per_group):
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                first_comp_node = True
                with (
                    self._open_output_stream(npu_id) as g,
                    self._open_output_stream(npu_id + self.num_npus) as s,
                ):
                    global_metadata = self.get_global_metadata()
                    self._emit_message(g, global_metadata)
                    self._emit_message(s, global_metadata)
                    if evict != None:
                        self._emit_message(g, evict)
                    if load != None:
                        self._emit_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        self._emit_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            self._emit_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            self._emit_message(g, receive_input_node)

                    expert_start = False
                    layer_num = layer_start
                    while expert_start or layer_num < layer_end:
                        if not layers[layer_num].is_expert:
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                self._emit_message(g, weight_load_node)
                            
                            # Compute
                            if layers[layer_num].comp_time != 0:
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name, 
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node

                                if first_comp_node:
                                    if npu_group == 0:
                                        self.add_parent(comp_node, input_load_node)
                                    else:
                                        self.add_parent(comp_node, receive_input_node)
                                    if evict != None:
                                        self.add_parent(comp_node, evict)
                                    if load != None:
                                        self.add_parent(comp_node, load)
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    first_comp_node = False
                                else:
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    if layers[layer_num - 1].comm_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                    else:
                                        self.add_parent(comp_node, layers[layer_num - 2].comp_node)
                                
                                self._emit_message(g, comp_node)

                                # Send KV cache after each kv_proj
                                if "v_proj" in layers[layer_num].name:
                                    send_kv_node = self.get_comm_node(
                                        is_send=True,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].output_memory_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus, # to the paired npu in decode
                                        id=layer_num
                                    )
                                    self.add_parent(send_kv_node, comp_node)
                                    self._emit_message(g, send_kv_node)

                                    recv_kv_node = self.get_comm_node(
                                        is_send=False,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].output_memory_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus,
                                        id=layer_num
                                    )
                                    self._emit_message(s, recv_kv_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                self._emit_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            if expert_start == False and use_comm:
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num-1].output_memory_size)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                self._emit_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                                # End of expert, add ALLTOALL communication after expert computation
                                if use_comm:
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num+1].input_memory_size)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    self._emit_message(g, comm_coll_node)
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1

                    # update new layer_end
                    layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Send output (for the last layer, to the paired decode npu)
                        send_output_node = self.get_comm_node(
                            is_send=True,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        self._emit_message(g, send_output_node)
                        # paired decode npu receive output
                        recv_output_node = self.get_comm_node(
                            is_send=False,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        self._emit_message(s, recv_output_node)
                    else:
                        if layers[layer_end - 1].is_expert:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        self._emit_message(g, send_output_node)
            if pipeline_boundaries is None:
                remain_layers -= 1

    def convert_event(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)
        for npu_id in range(self.num_npus):
            with self._open_output_stream(npu_id) as g:
                global_metadata = self.get_global_metadata()
                self._emit_message(g, global_metadata)
                for idx, layer in enumerate(layers):
                    comp_node = self.get_comp_node(
                    layer.name, 
                    layer.comp_time)
                    layer.comp_node = comp_node
                    self._emit_message(g, comp_node)

    def _convert(self) -> None:
        with self._open_input() as f:
            first_line = f.readline().strip().split()
            execution_type = first_line[0]

            pipeline_boundaries = None
            if len(first_line) >= 3:
                assert(first_line[1] == "model_parallel_NPU_group:")
                num_npu_group = int(first_line[2])
                if len(first_line) == 5:
                    assert(first_line[3] == "pipeline_block_boundaries:")
                    pipeline_boundaries = [int(value) for value in first_line[4].split(",")]
            else:
                num_npu_group = 0

            second_line = f.readline().strip()
            num_layers = int(second_line)

            third_line = f.readline() # This is for the table header, so just ignore it

            if execution_type == "COLOCATED":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, pipeline_boundaries)
            elif execution_type == "PREFILL":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_prefill(f, num_layers, num_npu_group, pipeline_boundaries)
            elif execution_type == "DECODE":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, pipeline_boundaries)
            elif execution_type == "EVENT":
                self.convert_event(f, num_layers)
            else:
                raise ValueError(f"Unsupported execution type, {execution_type}")

    def convert(self) -> None:
        """Convert the input trace to the legacy rank-specific ET files."""
        self._reset_conversion_state()
        self._payload_streams = None
        self._convert()

    def convert_to_payloads(self, compact_metadata: bool = False) -> Dict[int, bytes]:
        """Convert the input trace to rank-indexed ET payloads without files.

        Returned bytes are bit-for-bit equivalent to the legacy file contents
        produced from a fresh converter with the same inputs unless compact
        metadata is requested. Compact metadata stores a trace digest instead
        of duplicating the full source trace for every rank; ETFeeder does not
        use GlobalMetadata to simulate execution.
        """
        self._reset_conversion_state()
        self._payload_streams = {}
        self._compact_metadata = compact_metadata
        try:
            self._convert()
            return {
                npu_id: stream.getvalue()
                for npu_id, stream in self._payload_streams.items()
            }
        finally:
            self._payload_streams = None

    def convert_to_template_bundle(
        self, known_template_ids=None, compact_metadata: bool = False,
    ):
        """Convert directly to the shared-template protocol representation.

        Unlike ``convert_to_payloads`` this never materialises framed rank ET
        byte strings only to decode them again in the serving controller.
        """
        from inference_serving.execution_templates import TemplateBundleCollector

        self._reset_conversion_state()
        self._compact_metadata = compact_metadata
        self._template_collector = TemplateBundleCollector(known_template_ids)
        try:
            self._convert()
            return self._template_collector.build()
        finally:
            self._template_collector = None
