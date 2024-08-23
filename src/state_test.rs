use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

mod utils;

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
struct StateTestState {
    main_stage: U2VarSys,
    addr_step: UDynVarSys,
    step_stage: BoolVarSys,
    unused: BoolVarSys,
    value: UDynVarSys,
    iter_count: UDynVarSys,
    addr_step_num: usize,
}

impl StateTestState {
    fn new(data_part_len: u32, max_proc_num_bits: u32, value_bits: u32, iter_num: u64) -> Self {
        let data_part_len = data_part_len as usize;
        let addr_step_num = ((max_proc_num_bits as usize) + data_part_len - 1) / data_part_len;
        Self {
            main_stage: U2VarSys::var(),
            addr_step: UDynVarSys::var(calc_log_bits(addr_step_num)),
            step_stage: BoolVarSys::var(),
            unused: BoolVarSys::var(),
            value: UDynVarSys::var(value_bits as usize),
            iter_count: UDynVarSys::var(calc_log_bits_u64(iter_num)),
            addr_step_num,
        }
    }

    fn to_dynintvar(self) -> UDynVarSys {
        UDynVarSys::from(self.main_stage)
            .concat(self.addr_step)
            .concat(UDynVarSys::from_iter([self.step_stage, self.unused]))
            .concat(self.value)
            .concat(self.iter_count)
    }
}

fn gen_state_test(
    cell_len_bits: u32,
    data_part_len: u32,
    proc_num: u64,
    max_proc_num_bits: u32,
    value_bits: u32,
    iter_num: u64,
    int_iter_num: u64,
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

    let cell_len = 1 << cell_len_bits;
    let old_state = StateTestState::new(data_part_len, max_proc_num_bits, value_bits, iter_num);
    let addr_step_max =
        UDynVarSys::from_n(old_state.addr_step_num - 1, old_state.addr_step.bitnum());
    let iter_max = UDynVarSys::from_n(iter_num - 1, old_state.iter_count.bitnum());

    let unused_value = &old_state.unused
        | &mobj.in_dp_move_done
        | mobj
            .in_memval
            .iter()
            .fold(BoolVarSys::from(false), |a, x| a.clone() | x.clone());
    // 1. Load proc id to mem_address and to value
    let mut state_1 = old_state.clone();
    state_1.main_stage = int_ite(
        &old_state.step_stage & (&old_state.addr_step).equal(&addr_step_max),
        U2VarSys::from(1u32),
        U2VarSys::from(0u32),
    );
    state_1.addr_step = dynint_ite_r(
        &old_state.step_stage,
        &(&old_state.addr_step + 1u32),
        &old_state.addr_step,
    );
    state_1.step_stage = !&old_state.step_stage;
    // store proc id into state.value in given position.
    // use highest possible shift: addr_step_num * data_part_len
    let max_init_value_bits = old_state.addr_step_num * (data_part_len as usize);
    let mivalue_bits_bits = calc_log_bits(max_init_value_bits as usize);
    let old_dpval = UDynVarSys::try_from_n(mobj.in_dpval.clone(), 1 << mivalue_bits_bits).unwrap();
    let old_addr_step_ext =
        UDynVarSys::try_from_n(old_state.addr_step.clone(), mivalue_bits_bits).unwrap();
    let shifted_dpval = if value_bits as usize <= old_dpval.bitnum() {
        (old_dpval << (old_addr_step_ext * data_part_len)).subvalue(0, value_bits as usize)
    } else {
        UDynVarSys::try_from_n(
            old_dpval << (old_addr_step_ext * data_part_len),
            value_bits as usize,
        )
        .unwrap()
    };
    state_1.value = dynint_ite_r(
        &old_state.step_stage,
        &(&old_state.value | &shifted_dpval),
        &old_state.value,
    );
    state_1.unused = unused_value.clone();
    let mut mach_out_1 = InfParOutputSys::new(config);
    mach_out_1.state = state_1.to_dynintvar();
    mach_out_1.dpr = !&old_state.step_stage;
    mach_out_1.dpw = old_state.step_stage.clone();
    mach_out_1.dpval = mobj.in_dpval.clone();
    mach_out_1.dkind = int_ite(
        old_state.step_stage.clone(),
        U2VarSys::from(DKIND_MEM_ADDRESS),
        U2VarSys::from(DKIND_PROC_ID),
    );
    mach_out_1.dpmove = DPMOVE_FORWARD.into();

    // 2. Do calculations
    let mut state_2 = old_state.clone();
    state_2.main_stage = int_ite(
        (&old_state.iter_count).equal(iter_max),
        U2VarSys::from(2u32),
        U2VarSys::from(1u32),
    );
    state_2.iter_count = &old_state.iter_count + 1u8;
    let value_mask = if value_bits < 32 {
        (1u32 << value_bits) - 1u32
    } else {
        u32::MAX
    };
    state_2.value = old_state.value.clone();
    for _ in 0..int_iter_num {
        state_2.value = ((&state_2.value + (0x11aabcdu32 & value_mask))
            * (&state_2.value + (0xfa2135u32 & value_mask)))
            ^ &state_2.value;
    }
    state_2.unused = unused_value.clone();
    let mut mach_out_2 = InfParOutputSys::new(config);
    mach_out_2.state = state_2.to_dynintvar();

    // 3. Do write highest part of value to memory
    let mut state_3 = old_state.clone();
    state_3.main_stage = U2VarSys::from(2u32);
    state_3.unused = unused_value.clone();
    let mut mach_out_3 = InfParOutputSys::new(config);
    mach_out_3.state = state_3.to_dynintvar();
    mach_out_3.memw = BoolVarSys::from(true);
    mach_out_3.memval = if cell_len < value_bits as usize {
        old_state
            .value
            .clone()
            .subvalue((value_bits as usize) - cell_len, cell_len)
    } else {
        UDynVarSys::try_from_n(old_state.value.clone(), cell_len).unwrap()
    };
    mach_out_3.stop = BoolVarSys::from(true);

    // join
    let final_state = dynint_table(
        UDynVarSys::from(old_state.main_stage.clone()),
        [mach_out_1, mach_out_2, mach_out_3.clone(), mach_out_3]
            .into_iter()
            .map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(old_state.to_dynintvar());
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_state_test_expmem(
    cell_len_bits: u32,
    proc_num: u64,
    value_bits: u32,
    iter_num: u64,
    int_iter_num: u64,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let cell_len = 1 << cell_len_bits;
    let cell_len_in_bytes = if cell_len_bits >= 3 {
        1 << (cell_len_bits - 3)
    } else {
        1
    };
    let proc_cell_mask = if cell_len_bits < 3 {
        (1 << (3 - cell_len_bits)) - 1
    } else {
        0
    };
    let cell_mask = (1 << cell_len) - 1;
    let mut cell = 0u8;

    let mut file = BufWriter::new(fs::File::create(path)?);
    let value_mask = (1u128 << value_bits) - 1;

    for i in 0..proc_num {
        let mut value = (i as u128) & value_mask;
        for _ in 0..iter_num {
            for _ in 0..int_iter_num {
                value = (value + (0x11aabcdu128 & value_mask))
                    * (value + (0xfa2135u128 & value_mask))
                    ^ value;
            }
        }
        let out = if cell_len < value_bits as usize {
            value >> (value_bits as usize - cell_len)
        } else {
            value
        };
        let bytes = out.to_le_bytes();
        if cell_len_bits >= 3 {
            file.write(&bytes[0..cell_len_in_bytes])?;
        } else {
            cell |= u8::try_from((bytes[0] & cell_mask) << ((i & proc_cell_mask) << cell_len_bits))
                .unwrap();
            if (i & proc_cell_mask) == proc_cell_mask {
                // do write
                file.write(&[cell][0..1])?;
                cell = 0;
            }
        };
    }
    Ok(())
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let command = args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    let max_proc_num_bits: u32 = args.next().unwrap().parse().unwrap();
    let value_bits: u32 = args.next().unwrap().parse().unwrap();
    let iter_num: u64 = args.next().unwrap().parse().unwrap();
    let int_iter_num: u64 = args.next().unwrap().parse().unwrap();
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    assert_ne!(max_proc_num_bits, 0);
    assert!(max_proc_num_bits <= 64);
    assert!(u128::from(proc_num) <= (1u128 << max_proc_num_bits));
    assert_ne!(value_bits, 0);
    assert!(iter_num >= 2);
    assert!(int_iter_num >= 1);
    match command.as_str() {
        "machine" => {
            print!(
                "{}",
                callsys(|| gen_state_test(
                    cell_len_bits,
                    data_part_len,
                    proc_num,
                    max_proc_num_bits,
                    value_bits,
                    iter_num,
                    int_iter_num,
                )
                .unwrap())
            );
        }
        // expected memory
        "expmem" => {
            let path = args.next().unwrap();
            gen_state_test_expmem(
                cell_len_bits,
                proc_num,
                value_bits,
                iter_num,
                int_iter_num,
                path,
            )
            .unwrap()
        }
        _ => {
            panic!("Unknown command");
        }
    }
}
