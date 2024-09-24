use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

// Utilities for machine.
// General utitities for machine creation. Utilities designed to be generic and usable
// on machine with any value of parameters - cell_len_bits and data_part_len, proc_num...
// possible smallest cell_len_bits is 0 (cell_len=1), possible smallest data_part_len is 1.
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
//
// Temp buffer organization:
// data_part_len > 1: [XEND0,X0_0,X1_0,X2_0,....,XEND1,X0_1,X1_1,X2_1,.....,
//                     XEND2,X0_2,X1_2,X2_2......]
// XENDx - hold end position marker (end_pos):
//        bit 0 - memory end position marker, bit 1 - proc id end position marker.
//        If bit of given end marker is 1 - then no more data of given type of data starting
//        from this place.
// Xx_y - y'th data part of data x. Number of data parts will be determined by designer.
//
// data_part_len = 1: [MEND0,PEND0,X0_0,X1_0,X2_0,....,MEND1,PEND1,X0_1,X1_1,X2_1,.....,
//                     MEND2,PEND2,X0_2,X1_2,X2_2......]
// MENDx - hold memory end position marker. If bit is 1 then no more data in data starting
//        from this place.
// PENDx - hold proc_id end position marker. If bit is 1 then no more data in data starting
//        from this place.
// Xx_y - y'th data part of data x. Number of data parts will be determined by designer.
//
// temp_buffer_step - Number of datas including end position markers.
//
// Limiter: end position marker datas.
// Placed in first temp buffer data_part_len bit words.
// Temp buffer chunk part: [WORD0, WORD1,...]
// WORD0: 0 bit - memory address end pos, 1 bit - proc id end pos, 2 bit - other end pos, ....

// function parameters in infinite data (memaddrss, tempbuffer, procids).

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InfDataParam {
    MemAddress,
    ProcId,
    TempBuffer(u32),
    EndPos(u32),
}

// define default end position markers
pub const END_POS_MEM_ADDRESS: u32 = 0;
pub const END_POS_PROC_ID: u32 = 1;

pub mod basic;
pub use basic::*;
pub mod functions;
pub use functions::*;
pub mod infdata;
pub use infdata::*;
pub mod helpers;
pub use helpers::*;
