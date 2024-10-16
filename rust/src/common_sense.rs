// nested loop
fn nested_loop() {
    let input: String = "q".to_string();
    'loop_1: loop {
        println!("loop 1 start");
        'loop_2: loop {
            println!("loop 2");
            if input == "q" {
                break 'loop_1;
            }
            break 'loop_2;
        }
        println!("loop 1 end");
    }
}

// Tuple
fn tuple() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let tup1: (i8, i16, f32) = (1, 2, 3.14);
    let (x, y, z) = tup;

    println!("The value of x is: {}", x);
    println!("The value of y is: {}", y);
    println!("The value of z is: {}", z);

    println!("The value of tup1.0 is: {}", tup1.0);
    println!("The value of tup1.1 is: {}", tup1.1);
    println!("The value of tup1.2 is: {}", tup1.2);
}

// Array
fn array() {
  let test2: [i32;7] = [1,2,3,4,5,6,7];
  println!("The value of test2 is: {}", test2[0]);
  println!("The value of test2 is: {}", test2[1]);
  println!("The value of test2 is: {}", test2[2]);
  println!("The value of test2 is: {}", test2[3]);
  println!("The value of test2 is: {}", test2[4]);
  println!("The value of test2 is: {}", test2[5]);
  println!("The value of test2 is: {}", test2[6]);
  // Runtime error
  // println!("The value of test2 is: {}", test2[8]);
}

fn main() {
    nested_loop();
    tuple();
    array();
}
