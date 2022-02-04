use super::checkers::BrightnessChecker;
use super::hashers::{MeanBrightnessHasher, PixelHasher};
use super::*;
extern crate test;

#[test]
fn pixels_to_array_test() {
    assert_eq!(
        converters::pixels_to_array(&vec![42u8], 1),
        vec![vec![42u8]]
    );
    assert_eq!(
        converters::pixels_to_array(&vec![42u8; 9], 3),
        vec![
            vec![42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8]
        ]
    );
    assert_eq!(
        converters::pixels_to_array(&vec![42u8; 12], 3),
        vec![
            vec![42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8]
        ]
    );
    assert_eq!(
        converters::pixels_to_array(&vec![42u8; 12], 4),
        vec![
            vec![42u8, 42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8, 42u8],
            vec![42u8, 42u8, 42u8, 42u8]
        ]
    );
}

#[test]
fn brightness_hasher_test() {
    let b = MeanBrightnessHasher {};
    let arr: Vec<Vec<u8>> = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
        vec![10, 11, 12],
    ];
    assert_eq!(b.hash(&arr, Position::new(0, 0), Dimensions::new(4, 3)), 6);
    assert_eq!(b.hash(&arr, Position::new(1, 1), Dimensions::new(2, 2)), 7);
    assert_eq!(b.hash(&arr, Position::new(1, 2), Dimensions::new(1, 1)), 8);
}

#[bench]
fn open_discrete_rgb16(b: &mut test::Bencher) {
    let img = image::io::Reader::open("./test/test.jpg")
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .into_rgb16();
    let w = img.width();
    let v: Vec<[u16; 3]> = img.pixels().map(|f| f.0).collect();
    b.iter(move || {
        DiscreteImage::new(
            v.clone(),
            hashers::MeanBrightnessHasher {},
            w,
            BrightnessChecker {
                precision: [200u16, 200, 200],
            },
            10,
            100000,
        );
    });
}

#[bench]
fn open_discrete_luma16(b: &mut test::Bencher) {
    let img = image::io::Reader::open("./test/test.jpg")
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .into_luma16();
    let w = img.width();
    let v: Vec<u16> = img.pixels().map(|f| f.0[0]).collect();
    b.iter(move || {
        DiscreteImage::new(
            v.clone(),
            hashers::MeanBrightnessHasher {},
            w,
            BrightnessChecker { precision: 100u16 },
            10,
            100000,
        );
    });
}
