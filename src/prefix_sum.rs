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
    // 1. Load proc_id to mem address
    // 2. Move back position to start
    // 3. Read main value and to store value in state.
    // 4. Main algorithm. N iterations, i - itetation, N - proc_num_bits:
    // 4.1. Calculate addr_sub_bit: if (i > 0) { i - 1 } else { 0 }.
    // 4.2. Subtract (1 << addr_sub_bit) from memory address.
    // 4.2.1. Move position to addr_sub_bit / data_part_len.
    // 4.2.2. Subtract bit from mem_address_part and store mem_address_part.
    // 4.2.3. Subtract old carry from mem_address_part and store mem_address_part.
    // 4.2.4. Move back position to start
    // 4.3. If no carry ((1 << addr_sub_bit) < mem_addres) then skip all steps.
    // 4.4. Read value of this address.
    // 4.5. Add to state_value and store into state_value.
    // 5. Load proc_id to mem address
    // 6. Store state_value into memory.
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
