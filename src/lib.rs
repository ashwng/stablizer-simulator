//! A Rust implementation of Aaronson & Gottesman, *Improved Simulation of
//! Stabilizer Circuits* (Phys. Rev. A 70, 052328, 2004; quant-ph/0406196).
//!
//! | Paper section                           | Module                                       |
//! |-----------------------------------------|----------------------------------------------|
//! | Section II, Proposition 2               | [`counting`]                                 |
//! | Section III tableau algorithm           | [`tableau`] (on top of [`rows`], [`pauli`])  |
//! | Section III inner product               | [`inner_product`]                            |
//! | Section IV CHP language and experiments | [`circuit`], [`chp`], [`experiment`]         |
//! | Section V Theorem 4 (⊕L procedure)      | [`parity_l`]                                 |
//! | Section VI canonical form, Corollary 9  | [`canonical`], [`pmh`]                       |
//! | Section VII A mixed states              | [`mixed`]                                    |
//! | Section VII B non-stabilizer inputs     | [`nonstab_init`]                             |
//! | Section VII C non-stabilizer gates      | [`nonstab_gates`]                            |
//!
//! Qubits and rows are 0-indexed throughout (the paper is 1-indexed).

pub mod canonical;
pub mod ch_form;
pub mod chp;
pub mod circuit;
mod columns;
pub mod counting;
pub mod error;
pub mod experiment;
pub mod gf2;
pub mod inner_product;
pub mod mixed;
pub mod nonstab_gates;
pub mod nonstab_init;
pub mod parity_l;
pub mod pauli;
pub mod pmh;
pub mod rng;
pub mod rows;
pub mod tableau;

#[cfg(feature = "python")]
mod python;

pub use ch_form::ChForm;
pub use circuit::{Circuit, Gate};
pub use error::{Error, Result};
pub use pauli::{g, PauliString};
pub use rng::Rng;
pub use tableau::{MeasureResult, Tableau};
