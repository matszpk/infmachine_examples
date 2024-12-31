use gate_calc_log_bits::*;
use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;

fn gen_simple_inc(
    cell_len_bits: u32,
    proc_num_bits: u32,
    real_proc_num_bits: u32,
) -> Result<String, toml::ser::Error> {
    let config = InfParInterfaceConfig {
        cell_len_bits,
        data_part_len: 1 << cell_len_bits,
    };
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num: 1 << real_proc_num_bits,
            flat_memory: true,
            max_mem_size: Some((1 << (real_proc_num_bits + cell_len_bits)) >> 3),
            max_temp_buffer_len: 64,
        },
    );
    // Stages:
    // 1. Load proc_id to mem_address. N steps. Step stages:
    // 1.1. Read proc_id and move position forward.
    // 1.2. Write value to mem addres and move position forward.
    // 2. Read memory
    // 3. Increment and write memory.
    // 4. Move data part pos to start. N steps: Step stages:
    // 4.1. Move forward proc id position
    // 4.2. Move forward memory address position
    // main code
    let cell_len = 1usize << cell_len_bits;
    let main_stage = U2VarSys::var();
    let addr_step_num = ((proc_num_bits as usize) + cell_len - 1) / cell_len;
    let addr_step = UDynVarSys::var(calc_log_bits(addr_step_num));
    let addr_step_max = UDynVarSys::from_n(addr_step_num - 1, addr_step.bitnum());
    let step_stage = BoolVarSys::var(); // true - second stage

    let in_state = UDynVarSys::from(main_stage.clone())
        .concat(addr_step.clone())
        .concat(UDynVarSys::from_iter([step_stage.clone()]));

    fn to_mach_state(ms: U2VarSys, ads: UDynVarSys, ss: BoolVarSys) -> UDynVarSys {
        UDynVarSys::from(ms)
            .concat(ads)
            .concat(UDynVarSys::from_iter([ss]))
    }
    let addr_step_zero = UDynVarSys::from_n(0u32, addr_step.bitnum());
    // 1. Load proc id to mem_address
    let mut mach_out_1 = InfParOutputSys::new(config);
    mach_out_1.state = to_mach_state(
        int_ite(
            &step_stage & (&addr_step).equal(&addr_step_max),
            U2VarSys::from(1u32),
            U2VarSys::from(0u32),
        ),
        dynint_ite_r(&step_stage, &(&addr_step + 1u32), &addr_step),
        !&step_stage,
    );
    mach_out_1.dpr = !&step_stage;
    mach_out_1.dpw = step_stage.clone();
    mach_out_1.dpval = mobj.in_dpval.clone();
    mach_out_1.dkind = int_ite(
        step_stage.clone(),
        U2VarSys::from(DKIND_MEM_ADDRESS),
        U2VarSys::from(DKIND_PROC_ID),
    );
    mach_out_1.dpmove = DPMOVE_FORWARD.into();
    // 2. Read memory
    let mut mach_out_2 = InfParOutputSys::new(config);
    mach_out_2.state = to_mach_state(2u32.into(), addr_step_zero.clone(), false.into());
    mach_out_2.memr = BoolVarSys::from(true);
    mach_out_2.memw = BoolVarSys::from(false);
    // 3. Increment and write memory
    let mut mach_out_3 = InfParOutputSys::new(config);
    mach_out_3.state = to_mach_state(3u32.into(), addr_step_zero.clone(), false.into());
    mach_out_3.memr = BoolVarSys::from(false);
    mach_out_3.memw = BoolVarSys::from(true);
    mach_out_3.memval = &mobj.in_memval + 1u32;
    // 4. Move back positions
    let mut mach_out_4 = InfParOutputSys::new(config);
    mach_out_4.state = to_mach_state(3u32.into(), addr_step_zero.clone(), !&step_stage);
    mach_out_4.dkind = int_ite(
        step_stage.clone(),
        U2VarSys::from(DKIND_MEM_ADDRESS),
        U2VarSys::from(DKIND_PROC_ID),
    );
    mach_out_4.dpmove = DPMOVE_BACKWARD.into();
    mach_out_4.stop = &step_stage & !&mobj.in_dp_move_done;

    // join
    let final_state = dynint_table(
        UDynVarSys::from(main_stage),
        [mach_out_1, mach_out_2, mach_out_3, mach_out_4]
            .into_iter()
            .map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(in_state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let proc_num_bits: u32 = args.next().unwrap().parse().unwrap();
    let real_proc_num_bits: u32 = if let Some(arg) = args.next() {
        arg.parse().unwrap()
    } else {
        proc_num_bits
    };
    assert!(cell_len_bits <= 16);
    assert_ne!(proc_num_bits, 0);
    assert!((1 << cell_len_bits) < proc_num_bits);
    assert!(real_proc_num_bits <= proc_num_bits);
    assert!(real_proc_num_bits < 64);
    print!(
        "{}",
        callsys(|| gen_simple_inc(cell_len_bits, proc_num_bits, real_proc_num_bits).unwrap())
    );
}
