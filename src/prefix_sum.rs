use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;

const fn calc_log_bits(n: usize) -> usize {
    let nbits = usize::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

const fn calc_log_bits_u64(n: u64) -> usize {
    let nbits = u64::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

#[derive(Clone)]
struct PrefixSumState {
    main_stage: U2VarSys,
    addr_step: UDynVarSys,
    addr_step_bit: UDynVarSys,
    stage_step: BoolVarSys,
    value: UDynVarSys,
}

impl PrefixSumState {
    fn new(cell_len_bits: u32, data_part_len: u32, max_proc_num_bits: u32) -> Self {
        let addr_step_num = (max_proc_num_bits + data_part_len - 1) / data_part_len;
        let addr_step_len = calc_log_bits(addr_step_num as usize);
        let addr_step_bit_len = calc_log_bits(data_part_len as usize);
        Self {
            main_stage: U2VarSys::default(),
            addr_step: UDynVarSys::from_n(0u8, addr_step_len),
            addr_step_bit: UDynVarSys::from_n(0u8, addr_step_bit_len),
            stage_step: BoolVarSys::from(false),
            value: UDynVarSys::from_n(0u8, 1 << cell_len_bits),
        }
    }

    fn from_dynintvar(
        cell_len_bits: u32,
        data_part_len: u32,
        max_proc_num_bits: u32,
        state: UDynVarSys,
    ) -> Self {
        let addr_step_num = (max_proc_num_bits + data_part_len - 1) / data_part_len;
        let addr_step_len = calc_log_bits(addr_step_num as usize);
        let addr_step_bit_len = calc_log_bits(data_part_len as usize);
        let vars = state.subvalues(
            0,
            [2, addr_step_len, addr_step_bit_len, 1, 1 << cell_len_bits],
        );
        Self {
            main_stage: U2VarSys::try_from(vars[0].clone()).unwrap(),
            addr_step: vars[1].clone(),
            addr_step_bit: vars[2].clone(),
            stage_step: vars[3].bit(0),
            value: vars[3].clone(),
        }
    }

    fn to_dynintvar(self) -> UDynVarSys {
        UDynVarSys::from(self.main_stage)
            .concat(self.addr_step)
            .concat(self.addr_step_bit)
            .concat(UDynVarSys::filled(1, self.stage_step))
            .concat(self.value)
    }
}

fn gen_prefix_sum(
    cell_len_bits: u32,
    data_part_len: u32,
    proc_num: u64,
    max_proc_num_bits: u32,
) -> Result<String, toml::ser::Error> {
    let config = InfParInterfaceConfig {
        cell_len_bits,
        data_part_len,
    };
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num,
            flat_memory: true,
            max_mem_size: Some(((proc_num << cell_len_bits) + 7) >> 3),
            max_temp_buffer_len: max_proc_num_bits,
        },
    );
    // Main stages:
    // no_first = 0 - in state.
    // 0. Init memory and proc end pos.
    // 1. Move mem data to start.
    // 2. Initialize memory address = proc_id
    // 3. Initialize temp_buffer[sub] = 1 and state_carry = 1.
    // 4. Load data from memory.
    // 5. Do: mem_address = mem_address - temp_buffer[sub]
    //    if carry (if mem_address >= temp_buffer[first])
    //    state_carry &= carry
    // 6. Load memory data to state (arg1)
    // 7. If state_carry: cell = cell + arg1.
    // 8. If not no_first: temp_buffer[sub] <<= 1.
    // 9.  Set no_first = 1.
    //     Check if temp_buffer[sub] = end: if yes then: go to 10 otherwise go to 4.
    // 10. Initialize memory address = proc_id
    // 11. Store cell to mem_address.
    mobj.to_machine().to_toml()
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    let max_proc_num_bits: u32 = if let Some(arg) = args.next() {
        arg.parse().unwrap()
    } else {
        u32::try_from(calc_log_bits_u64(proc_num)).unwrap()
    };
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    assert_ne!(max_proc_num_bits, 0);
    assert!((1 << cell_len_bits) < max_proc_num_bits);
    assert!(max_proc_num_bits <= 64);
    assert!(u128::from(proc_num) <= (1u128 << max_proc_num_bits));
    print!(
        "{}",
        callsys(
            || gen_prefix_sum(cell_len_bits, data_part_len, proc_num, max_proc_num_bits).unwrap()
        )
    );
}
