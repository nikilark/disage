#![feature(test)]

pub mod pixels;

pub mod hashers;

pub mod converters;

pub mod discrete_image;

#[allow(unused_imports)]
pub use crate::discrete_image::*;

#[cfg(test)]
mod tests;
