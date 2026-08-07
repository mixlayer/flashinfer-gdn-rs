#![deny(unsafe_op_in_unsafe_fn)]
//! Candle CUDA tensor integration for FlashInfer GDN kernels.
//!
//! This crate is currently a workspace scaffold. It will adapt Candle storage,
//! layouts, allocations, and streams to the framework-independent APIs in
//! `flashinfer-gdn`.
