use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

// Utilities for machine.
// General utitities for machine creation. Utilities designed to be generic and usable
// on machine with any value of parameters - cell_len_bits and data_part_len, proc_num...
// possible smallest cell_len_bits is 1 (cell_len=2), possible smallest data_part_len is 2.
//
// Basic utilities:
// * load and determine max position at proc_id.
// * move back to start position in any data.
// * move data to another data (example: proc_id to mem_address or temp_buffer).
// * process data with any function.
// * any data operation includes stride (number of movements to next data part).
// * process integer in memory in ENDFORM form: [NP0, END0, NP1, END1, NP2, END2, ..].
//
// Number in ENDFORM: [NP0, END0, NP1, END1, NP2, END2, ..].
// NPx - part of number from lowest to highest.
// ENDx - mark last number part if value is not zero.
// Form: configurable circuit stage with specified part of stage indicator.
//       [STAGE_INDICATOR, STAGE_STATE, UNUSED]
// END_DP_FORM: [NP0_0, NP0_1, .., NP0_T, END0, NP1_0, NP1_1, .., NP1_T, END1, .....].
// T - (data_part_len + cell_len - 1) / cell_len.  Number of required cells to store data part.

// InitMemAddressEndPosState - initialize memory address end position from memory.
// Information about MemAddressEndPos in memory:
// At memory address 0: sequences of MAX (1<<cell_len) value cells and cell with value
// equal to 0 value.
// MemAddressPosEndPos is sum of all these cells plus number of these cells minus 1.
// Example: [3, 3, 3, 2, 0] - MemAddressPosEndPos is 15. [2, 0] - MemAddressPosEndPos is 3.
// Example: [3, 3, 3, 0] - MemAddressPosEndPos is 12. [1, 0] - MemAddressPosEndPos is 2.

#[derive(Clone)]
pub struct InitMemAddressEndPosStage {
    stage: UDynVarSys,   // stage indicator
    int_stage: U3VarSys, // internal stage indicator - only for this stage.
    state: U2VarSys,     // state for stage
}

impl InitMemAddressEndPosStage {
    pub fn new(cell_len_bits: usize, data_part_len: usize, stage_len: usize) -> Self {
        Self {
            stage: UDynVarSys::var(stage_len),
            int_stage: U3VarSys::var(),
            state: U2VarSys::var(),
        }
    }
}

// InitProcIdEndPosState - initialize proc id end position from memory.
// Information about ProcIdEndPos in memory: This same as in MemAddressEndPos and start
// after MemAddressEndPos in memory.

#[derive(Clone)]
pub struct InitProcIdEndPosStage {
    stage: UDynVarSys,   // stage indicator
    int_stage: U3VarSys, // internal stage indicator - only for this stage.
    state: U2VarSys,     // state for stage
}

impl InitProcIdEndPosStage {
    pub fn new(stage_len: usize) -> Self {
        Self {
            stage: UDynVarSys::var(stage_len),
            int_stage: U3VarSys::var(),
            state: U2VarSys::var(),
        }
    }
}
