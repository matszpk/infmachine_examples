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
struct StateTestState {
    main_stage: U2VarSys,
    addr_step: UDynVarSys,
    step_stage: BoolVarSys,
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
            value: UDynVarSys::var(value_bits as usize),
            iter_count: UDynVarSys::var(calc_log_bits_u64(iter_num)),
            addr_step_num,
        }
    }

    fn len(&self) -> usize {
        2 + self.addr_step.bitnum() + 1 + self.value.bitnum() + self.iter_count.bitnum()
    }

    fn to_dyntintvar(self) -> UDynVarSys {
        UDynVarSys::from(self.main_stage)
            .concat(self.addr_step)
            .concat(UDynVarSys::from_iter([self.step_stage]))
            .concat(self.value)
            .concat(self.iter_count)
    }

    fn from_dynintvar(&self, state: UDynVarSys) -> Self {
        let vars = state.subvalues(
            0,
            [
                2,
                self.addr_step.bitnum(),
                1,
                self.value.bitnum(),
                self.iter_count.bitnum(),
            ],
        );
        Self {
            main_stage: U2VarSys::try_from(vars[0].clone()).unwrap(),
            addr_step: vars[1].clone(),
            step_stage: vars[2].bit(0),
            value: vars[3].clone(),
            iter_count: vars[4].clone(),
            addr_step_num: self.addr_step_num,
        }
    }
}

fn gen_state_test(
    cell_len_bits: u32,
    data_part_len: u32,
    proc_num: u64,
    max_proc_num_bits: u32,
    value_bits: u32,
    iter_num: u64,
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

    let old_state = StateTestState::new(data_part_len, max_proc_num_bits, value_bits, iter_num);
    let addr_step_max =
        UDynVarSys::from_n(old_state.addr_step_num - 1, old_state.addr_step.bitnum());
    // 1. Load proc id to mem_address
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
    state_1.value = dynint_ite_r(
        &old_state.step_stage,
        &(&state_1.value
            | (UDynVarSys::try_from_n(mobj.in_dpval.clone(), value_bits as usize).unwrap()
                << (old_state.addr_step * data_part_len))),
        &state_1.value,
    );
    let mut mach_out_1 = InfParOutputSys::new(config);
    mach_out_1.state = state_1.to_dyntintvar();
    mach_out_1.dpr = !&old_state.step_stage;
    mach_out_1.dpw = old_state.step_stage.clone();
    mach_out_1.dpval = mobj.in_dpval.clone();
    mach_out_1.dkind = int_ite(
        old_state.step_stage.clone(),
        U2VarSys::from(DKIND_MEM_ADDRESS),
        U2VarSys::from(DKIND_PROC_ID),
    );
    mach_out_1.dpmove = DPMOVE_FORWARD.into();
    mobj.to_machine().to_toml()
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    let max_proc_num_bits: u32 = args.next().unwrap().parse().unwrap();
    let value_bits: u32 = args.next().unwrap().parse().unwrap();
    let iter_num: u64 = args.next().unwrap().parse().unwrap();
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    assert_ne!(max_proc_num_bits, 0);
    assert!((1 << cell_len_bits) < max_proc_num_bits);
    assert!(max_proc_num_bits <= 64);
    assert!(u128::from(proc_num) <= (1u128 << max_proc_num_bits));
    assert_ne!(value_bits, 0);
    assert_ne!(iter_num, 0);
    print!(
        "{}",
        callsys(|| gen_state_test(
            cell_len_bits,
            data_part_len,
            proc_num,
            max_proc_num_bits,
            value_bits,
            iter_num,
        )
        .unwrap())
    );
}
