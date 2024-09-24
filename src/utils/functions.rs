use std::ops::{BitAnd, BitOr, BitXor};

use super::*;

// functions

pub trait Function0 {
    fn state_len(&self) -> usize;
    // return (output state, output, external_outputs)
    fn output(&self, input_state: UDynVarSys) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>);
}

pub trait Function1 {
    fn state_len(&self) -> usize;
    // return (output state, output, external_outputs)
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>);
}

pub trait Function2 {
    fn state_len(&self) -> usize;
    // return (output state, output, external_outputs)
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>);
}

pub trait Function2_2 {
    fn state_len(&self) -> usize;
    // return (output state, output0, output1, external_outputs)
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, UDynVarSys, Vec<UDynVarSys>);
}

pub trait FunctionNN {
    fn state_len(&self) -> usize;
    fn input_num(&self) -> usize;
    fn output_num(&self) -> usize;
    // return (output state, outputs_vec, external_outputs)
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>);
}

pub struct FuncNNAdapter0<F: Function0> {
    f: F,
}

impl<F: Function0> From<F> for FuncNNAdapter0<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

pub struct FuncNNAdapter1<F: Function1> {
    f: F,
}

impl<F: Function1> From<F> for FuncNNAdapter1<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

pub struct FuncNNAdapter2<F: Function2> {
    f: F,
}

impl<F: Function2> From<F> for FuncNNAdapter2<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

pub struct FuncNNAdapter2_2<F: Function2_2> {
    f: F,
}

impl<F: Function2_2> From<F> for FuncNNAdapter2_2<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

impl<F: Function0> FunctionNN for FuncNNAdapter0<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        0
    }
    fn output_num(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        _: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (out_state, output, ext_outputs) = self.f.output(input_state);
        (out_state, vec![output], ext_outputs)
    }
}

impl<F: Function1> FunctionNN for FuncNNAdapter1<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        1
    }
    fn output_num(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (out_state, output, ext_outputs) = self.f.output(input_state, inputs[0].clone());
        (out_state, vec![output], ext_outputs)
    }
}

impl<F: Function2> FunctionNN for FuncNNAdapter2<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        2
    }
    fn output_num(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (out_state, output, ext_outputs) =
            self.f
                .output(input_state, inputs[0].clone(), inputs[1].clone());
        (out_state, vec![output], ext_outputs)
    }
}

impl<F: Function2_2> FunctionNN for FuncNNAdapter2_2<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        2
    }
    fn output_num(&self) -> usize {
        2
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (out_state, output, output2, ext_outputs) =
            self.f
                .output(input_state, inputs[0].clone(), inputs[1].clone());
        (out_state, vec![output, output2], ext_outputs)
    }
}

pub struct Zero0Func {
    inout_len: usize,
}

impl Zero0Func {
    pub fn new(inout_len: usize) -> Self {
        Self { inout_len }
    }
}

impl Function0 for Zero0Func {
    fn state_len(&self) -> usize {
        0
    }
    fn output(&self, _: UDynVarSys) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        (
            UDynVarSys::var(0),
            UDynVarSys::from_n(0u8, self.inout_len),
            vec![],
        )
    }
}

pub struct One0Func {
    inout_len: usize,
}

impl One0Func {
    pub fn new(inout_len: usize) -> Self {
        Self { inout_len }
    }
}

impl Function0 for One0Func {
    fn state_len(&self) -> usize {
        1
    }
    fn output(&self, state: UDynVarSys) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        (
            UDynVarSys::from_n(1u8, 1),
            dynint_ite(
                !state.bit(0),
                UDynVarSys::from_n(1u8, self.inout_len),
                UDynVarSys::from_n(0u8, self.inout_len),
            ),
            vec![],
        )
    }
}

pub struct Copy1Func {}

impl Copy1Func {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function1 for Copy1Func {
    fn state_len(&self) -> usize {
        0
    }
    fn output(&self, _: UDynVarSys, i0: UDynVarSys) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        (UDynVarSys::var(0), i0, vec![])
    }
}

// functions 1: func(arg1) = dest
// Bitwise operations

macro_rules! macro_bit1func {
    ($name:ident,$op:ident) => {
        pub struct $name {
            inout_len: usize,
            value: UDynVarSys,
        }

        impl $name {
            pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
                Self { inout_len, value }
            }
            pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
                Self {
                    inout_len,
                    value: if value != 0 {
                        UDynVarSys::from_n(value, (u64::BITS - value.leading_zeros()) as usize)
                    } else {
                        UDynVarSys::from_n(value, 1)
                    },
                }
            }
        }

        impl Function1 for $name {
            fn state_len(&self) -> usize {
                calc_log_bits(((self.value.bitnum() + self.inout_len - 1) / self.inout_len) + 1)
            }
            fn output(
                &self,
                input_state: UDynVarSys,
                i0: UDynVarSys,
            ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
                let max_state_count = (self.value.bitnum() + self.inout_len - 1) / self.inout_len;
                let state_len = self.state_len();
                // get current part of value to add to input.
                let index = input_state.clone().subvalue(0, state_len);
                let arg2 = dynint_table_partial(
                    index.clone(),
                    (0..max_state_count).map(|i| {
                        UDynVarSys::try_from_n(
                            self.value.subvalue(
                                i * self.inout_len,
                                std::cmp::min((i + 1) * self.inout_len, self.value.bitnum())
                                    - i * self.inout_len,
                            ),
                            self.inout_len,
                        )
                        .unwrap()
                    }),
                    UDynVarSys::from_n(0u8, self.inout_len),
                );
                let result = i0.$op(arg2);
                let next_state = dynint_ite(
                    (&index).equal(max_state_count),
                    UDynVarSys::from_n(max_state_count, state_len),
                    &index + 1u8,
                );
                (next_state, result, vec![])
            }
        }
    };
}

macro_bit1func!(And1Func, bitand);
macro_bit1func!(Or1Func, bitor);
macro_bit1func!(Xor1Func, bitxor);

// Add1Func
pub struct Add1Func {
    inout_len: usize,
    value: UDynVarSys,
    sign: BoolVarSys,
}

impl Add1Func {
    pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
        Self {
            inout_len,
            value,
            sign: false.into(),
        }
    }
    pub fn new_signed(inout_len: usize, value: IDynVarSys) -> Self {
        let sign = value.bit(value.bitnum() - 1);
        Self {
            inout_len,
            value: value.as_unsigned(),
            sign,
        }
    }
    pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
        Self {
            inout_len,
            value: if value != 0 {
                UDynVarSys::from_n(value, (u64::BITS - value.leading_zeros()) as usize)
            } else {
                UDynVarSys::from_n(value, 1)
            },
            sign: false.into(),
        }
    }
    pub fn new_from_i64(inout_len: usize, value: i64) -> Self {
        let abs_value = value.abs();
        Self {
            inout_len,
            value: if abs_value != 0 {
                let bits = (u64::BITS - abs_value.leading_zeros()) as usize;
                let mask = if bits < 64 {
                    (1u64 << bits) - 1
                } else {
                    u64::MAX
                };
                UDynVarSys::from_n((value as u64) & mask, bits)
            } else {
                UDynVarSys::from_n(value as u64, 1)
            },
            sign: (value < 0).into(),
        }
    }
}

impl Function1 for Add1Func {
    fn state_len(&self) -> usize {
        calc_log_bits(((self.value.bitnum() + self.inout_len - 1) / self.inout_len) + 1) + 1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        let max_state_count = (self.value.bitnum() + self.inout_len - 1) / self.inout_len;
        let state_len = self.state_len();
        // get current part of value to add to input.
        let index = input_state.clone().subvalue(0, state_len - 1);
        let old_carry = input_state.bit(state_len - 1);
        let adder = dynint_table_partial(
            index.clone(),
            (0..max_state_count).map(|i| {
                let part = self.value.subvalue(
                    i * self.inout_len,
                    std::cmp::min((i + 1) * self.inout_len, self.value.bitnum())
                        - i * self.inout_len,
                );
                let part_len = part.bitnum();
                if part_len < self.inout_len {
                    part.concat(UDynVarSys::filled(
                        self.inout_len - part_len,
                        self.sign.clone(),
                    ))
                } else {
                    part
                }
            }),
            UDynVarSys::filled(self.inout_len, self.sign.clone()),
        );
        let (result, carry) = i0.addc_with_carry(&adder, &old_carry);
        let next_state = dynint_ite(
            (&index).equal(max_state_count),
            UDynVarSys::from_n(max_state_count, state_len - 1),
            &index + 1u8,
        )
        .concat(UDynVarSys::filled(1, carry.clone()));
        (next_state, result, vec![UDynVarSys::filled(1, carry)])
    }
}

// Sub1Func
pub struct Sub1Func {
    inout_len: usize,
    value: UDynVarSys,
    sign: BoolVarSys,
}

impl Sub1Func {
    pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
        Self {
            inout_len,
            value,
            sign: false.into(),
        }
    }
    pub fn new_signed(inout_len: usize, value: IDynVarSys) -> Self {
        let sign = value.bit(value.bitnum() - 1);
        Self {
            inout_len,
            value: value.as_unsigned(),
            sign,
        }
    }
    pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
        Self {
            inout_len,
            value: if value != 0 {
                UDynVarSys::from_n(value, (u64::BITS - value.leading_zeros()) as usize)
            } else {
                UDynVarSys::from_n(value, 1)
            },
            sign: false.into(),
        }
    }
    pub fn new_from_i64(inout_len: usize, value: i64) -> Self {
        let abs_value = value.abs();
        Self {
            inout_len,
            value: if abs_value != 0 {
                let bits = (u64::BITS - abs_value.leading_zeros()) as usize;
                let mask = if bits < 64 {
                    (1u64 << bits) - 1
                } else {
                    u64::MAX
                };
                UDynVarSys::from_n((value as u64) & mask, bits)
            } else {
                UDynVarSys::from_n(value as u64, 1)
            },
            sign: (value < 0).into(),
        }
    }
}

impl Function1 for Sub1Func {
    fn state_len(&self) -> usize {
        calc_log_bits(((self.value.bitnum() + self.inout_len - 1) / self.inout_len) + 1) + 1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        let max_state_count = (self.value.bitnum() + self.inout_len - 1) / self.inout_len;
        let state_len = self.state_len();
        // get current part of value to add to input.
        let index = input_state.clone().subvalue(0, state_len - 1);
        let old_carry = input_state.bit(state_len - 1);
        let adder = dynint_table_partial(
            index.clone(),
            (0..max_state_count).map(|i| {
                let part = self.value.subvalue(
                    i * self.inout_len,
                    std::cmp::min((i + 1) * self.inout_len, self.value.bitnum())
                        - i * self.inout_len,
                );
                let part_len = part.bitnum();
                if part_len < self.inout_len {
                    part.concat(UDynVarSys::filled(
                        self.inout_len - part_len,
                        self.sign.clone(),
                    ))
                } else {
                    part
                }
            }),
            UDynVarSys::filled(self.inout_len, self.sign.clone()),
        );
        let (result, carry) = i0.addc_with_carry(&!adder, &!old_carry);
        let next_state = dynint_ite(
            (&index).equal(max_state_count),
            UDynVarSys::from_n(max_state_count, state_len - 1),
            &index + 1u8,
        )
        .concat(UDynVarSys::filled(1, !carry.clone()));
        (next_state, result, vec![UDynVarSys::filled(1, carry)])
    }
}

// Mul1Func
pub struct Mul1Func {
    inout_len: usize,
    value: UDynVarSys,
}

impl Mul1Func {
    pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
        Self { inout_len, value }
    }
    pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
        Self {
            inout_len,
            value: if value != 0 {
                UDynVarSys::from_n(value, (u64::BITS - value.leading_zeros()) as usize)
            } else {
                UDynVarSys::from_n(value, 1)
            },
        }
    }
}

impl Function1 for Mul1Func {
    fn state_len(&self) -> usize {
        self.value.bitnum()
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        let value_len = self.value.bitnum();
        let part_num = (value_len + self.inout_len - 1) / self.inout_len;
        let mut mults = (0..part_num + 1)
            .map(|i| {
                let part_len = std::cmp::min(
                    self.inout_len,
                    self.inout_len + value_len - i * self.inout_len,
                );
                UDynVarSys::from_n(0u8, part_len)
            })
            .collect::<Vec<_>>();
        // make multiply
        for i in 0..part_num {
            let part_len = std::cmp::min(self.inout_len, value_len - i * self.inout_len);
            let argb = UDynVarSys::try_from_n(
                UDynVarSys::from_iter(
                    (0..part_len).map(|j| self.value.bit(self.inout_len * i + j)),
                ),
                self.inout_len,
            )
            .unwrap();
            // mul - product + previous element from mul.
            let mul = (&i0).fullmul(argb)
                + UDynVarSys::try_from_n(
                    mults[i].clone().concat(mults[i + 1].clone()),
                    self.inout_len << 1,
                )
                .unwrap();
            mults[i] = mul.clone().subvalue(0, self.inout_len);
            mults[i + 1] = mul.clone().subvalue(self.inout_len, part_len);
        }
        let (result, next_state) = (input_state.concat(UDynVarSys::from_n(0u8, self.inout_len))
            + UDynVarSys::from_iter(mults.iter().map(|m| m.iter()).flatten()))
        .split(self.inout_len);
        (next_state.clone(), result, vec![next_state])
    }
}

// Shl1Func - shift left - multiply by 2^n.
pub struct Shl1Func {
    inout_len: usize,
    shift: usize,
}

impl Shl1Func {
    pub fn new(inout_len: usize, shift: usize) -> Self {
        Self { inout_len, shift }
    }
}

impl Function1 for Shl1Func {
    fn state_len(&self) -> usize {
        self.shift
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        if self.shift <= self.inout_len {
            let state = i0.clone().subvalue(self.inout_len - self.shift, self.shift);
            (
                state.clone(),
                input_state.concat(i0.subvalue(0, self.inout_len - self.shift)),
                vec![state],
            )
        } else {
            let state = input_state
                .clone()
                .subvalue(self.inout_len, self.shift - self.inout_len)
                .concat(i0.clone());
            (
                state.clone(),
                input_state.clone().subvalue(0, self.inout_len),
                vec![state],
            )
        }
    }
}

// Mul1Func
pub struct MulAdd1Func {
    mul: Mul1Func,
    add: Add1Func,
}

impl MulAdd1Func {
    pub fn new(inout_len: usize, mul_val: UDynVarSys, add_val: UDynVarSys) -> Self {
        Self {
            mul: Mul1Func::new(inout_len, mul_val),
            add: Add1Func::new(inout_len, add_val),
        }
    }
    pub fn new_from_u64(inout_len: usize, mul_val: u64, add_val: u64) -> Self {
        Self {
            mul: Mul1Func::new_from_u64(inout_len, mul_val),
            add: Add1Func::new_from_u64(inout_len, add_val),
        }
    }
}

impl Function1 for MulAdd1Func {
    fn state_len(&self) -> usize {
        self.mul.state_len() + self.add.state_len()
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        let (mul_next_state, mul_result, mul_ext_outputs) = self
            .mul
            .output(input_state.clone().subvalue(0, self.mul.state_len()), i0);
        let (add_next_state, add_result, add_ext_outputs) = self.add.output(
            input_state
                .clone()
                .subvalue(self.mul.state_len(), self.add.state_len()),
            mul_result,
        );
        let mut ext_outputs = mul_ext_outputs;
        ext_outputs.extend(add_ext_outputs);
        (
            mul_next_state.concat(add_next_state),
            add_result,
            ext_outputs,
        )
    }
}

// Alignment

pub struct Align1Func {
    inout_len: usize,
    bits: u64,
}

impl Align1Func {
    pub fn new(inout_len: usize, bits: u64) -> Self {
        Self { inout_len, bits }
    }
}

impl Function1 for Align1Func {
    fn state_len(&self) -> usize {
        calc_log_bits_u64(self.bits / (self.inout_len as u64) + 2) + 2
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        // part_num - parts number except last with not fully filled.
        let part_num = self.bits / (self.inout_len as u64);
        let last_part_len = self.bits % (self.inout_len as u64);
        let counter_len = self.state_len() - 2;
        let (counter, rest) = input_state.split(counter_len);
        let inc = rest.bit(0);
        let carry = rest.bit(1);
        let new_counter = dynint_ite(
            (&counter).less_than(part_num + 1),
            &counter + 1u8,
            counter.clone(),
        );
        // new_or - true if some bit is 1 from i0.
        let new_or = bool_ite(
            (&counter).less_than(part_num),
            // collect bits to OR sum.
            i0.iter().fold(BoolVarSys::from(false), |a, x| a | x),
            // if counter==part_num then get last bits from last part.
            // because later this value is ignored then value calculated any way.
            i0.iter()
                .take(usize::try_from(last_part_len).unwrap())
                .fold(BoolVarSys::from(false), |a, x| a | x),
        );
        // if one bit 1 from bits less than 'bits'.
        let new_inc: BoolVarSys = new_or | inc;
        // new_i0 - filtered i0 - zeroing bits lower than 'bits'.
        let new_i0 = dynint_ite(
            (&counter).less_than(part_num),
            UDynVarSys::from_n(0u8, self.inout_len),
            dynint_ite(
                (&counter).equal(part_num),
                UDynVarSys::from_iter((0..self.inout_len).map(|i| {
                    if i as u64 >= last_part_len {
                        i0.bit(i)
                    } else {
                        false.into()
                    }
                })),
                i0.clone(),
            ),
        );
        let (result, new_carry) = new_i0.addc_with_carry(
            // get value to add - 2**(bits - part_num*inout_len)
            &dynint_ite(
                (&counter).equal(part_num),
                UDynVarSys::from_iter((0..self.inout_len).map(|i| {
                    if i as u64 == last_part_len {
                        new_inc.clone()
                    } else {
                        false.into()
                    }
                })),
                UDynVarSys::from_n(0u8, self.inout_len),
            ),
            &carry,
        );
        (
            new_counter.concat(UDynVarSys::from_iter([new_inc, new_carry.clone()])),
            result,
            vec![UDynVarSys::filled(1, new_carry)],
        )
    }
}

// functions 2: func(arg1, arg2) = dest
// Bit ops

macro_rules! macro_bit2func {
    ($name:ident, $op:ident) => {
        pub struct $name {}

        impl $name {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl Function2 for $name {
            fn state_len(&self) -> usize {
                0
            }
            fn output(
                &self,
                _: UDynVarSys,
                i0: UDynVarSys,
                i1: UDynVarSys,
            ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
                // get current part of value to add to input.
                let result = i0.$op(i1);
                (UDynVarSys::var(0), result, vec![])
            }
        }
    };
}

macro_bit2func!(And2Func, bitand);
macro_bit2func!(Or2Func, bitor);
macro_bit2func!(Xor2Func, bitxor);

// Add2Func
pub struct Add2Func {}

impl Add2Func {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function2 for Add2Func {
    fn state_len(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        // get current part of value to add to input.
        let old_carry = input_state.bit(0);
        let (result, carry) = i0.addc_with_carry(&i1, &old_carry);
        let next_state = UDynVarSys::filled(1, carry);
        (next_state.clone(), result, vec![next_state])
    }
}

// Sub2Func
pub struct Sub2Func {}

impl Sub2Func {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function2 for Sub2Func {
    fn state_len(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        // get current part of value to sub to input.
        let old_neg_carry = !input_state.bit(0);
        // start with carry=1 and negate argument i1.
        let (result, carry) = i0.addc_with_carry(&!i1, &old_neg_carry);
        let next_state = UDynVarSys::filled(1, !&carry);
        (next_state, result, vec![UDynVarSys::filled(1, carry)])
    }
}

//

pub struct Copy1NFunc {
    n: usize,
}

impl Copy1NFunc {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl FunctionNN for Copy1NFunc {
    fn state_len(&self) -> usize {
        0
    }
    fn input_num(&self) -> usize {
        1
    }
    fn output_num(&self) -> usize {
        self.n
    }
    fn output(
        &self,
        _: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        (
            UDynVarSys::var(0),
            (0..self.n).map(|_| inputs[0].clone()).collect::<Vec<_>>(),
            vec![],
        )
    }
}

//

pub struct Swap2Func {}

impl Swap2Func {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function2_2 for Swap2Func {
    fn state_len(&self) -> usize {
        0
    }
    fn output(
        &self,
        _: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, UDynVarSys, Vec<UDynVarSys>) {
        (UDynVarSys::var(0), i1, i0, vec![])
    }
}

//

pub struct XorNNFuncSample {
    inout_len: usize,
    input_num: usize,
    output_num: usize,
}

impl XorNNFuncSample {
    pub fn new(inout_len: usize, input_num: usize, output_num: usize) -> Self {
        Self {
            inout_len,
            input_num,
            output_num,
        }
    }
}
impl FunctionNN for XorNNFuncSample {
    fn state_len(&self) -> usize {
        0
    }
    fn input_num(&self) -> usize {
        self.input_num
    }
    fn output_num(&self) -> usize {
        self.output_num
    }
    fn output(
        &self,
        _: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let mut outputs = (0..self.output_num)
            .map(|_| UDynVarSys::from_n(0u8, self.inout_len))
            .collect::<Vec<_>>();
        for i in 0..self.input_num {
            outputs[i % self.output_num] ^=
                UDynVarSys::try_from_n(inputs[i].clone(), self.inout_len).unwrap();
        }
        (UDynVarSys::var(0), outputs, vec![])
    }
}
