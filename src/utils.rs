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

const fn calc_log_bits(n: usize) -> usize {
    let nbits = usize::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

pub fn extend_output_state(
    state_start: usize,
    extra_bits: usize,
    output: &InfParOutputSys,
) -> UDynVarSys {
    assert!(state_start <= output.state.bitnum());
    if state_start + extra_bits > output.state.bitnum() {
        output.state.clone().concat(UDynVarSys::var(
            state_start + extra_bits - output.state.bitnum(),
        ))
    } else {
        output.state.clone()
    }
}

// init_mem_address_end_pos - initialize memory address end position from memory.
// Information about MemAddressEndPos in memory:
// At memory address 0: sequences of values between 1..=MAX and one zero,
// MemAddressPosEndPos is sum of non-zero cells.
// If cell_len=1 then: sequences of 1 and one zero. MemAddressPosEndPos is number of 1's.

pub fn init_mem_address_end_pos_stage(
    state_start: usize,
    output: &InfParOutputSys,
) -> (UDynVarSys, InfParOutputSys) {
    (extend_output_state(state_start, 4, output), output.clone())
}

// init_proc_id_end_pos - initialize proc id end position from memory.
// Information about ProcIdEndPos in memory: This same as in MemAddressEndPos and start
// after MemAddressEndPos in memory.

pub fn init_proc_id_end_pos_stage(
    state_start: usize,
    output: &InfParOutputSys,
) -> (UDynVarSys, InfParOutputSys) {
    (extend_output_state(state_start, 4, output), output.clone())
}
