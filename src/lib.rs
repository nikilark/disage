#![feature(test)]

pub mod pixels;

pub mod hashers;

pub mod converters;

pub mod checkers;

pub mod discrete_image;

pub mod open;

#[allow(unused_imports)]
pub use crate::discrete_image::*;

#[allow(unused_imports)]
pub use crate::open::*;

#[cfg(test)]
mod tests;
