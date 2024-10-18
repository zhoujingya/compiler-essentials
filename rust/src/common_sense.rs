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
    let test2: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
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

// Struct
#[derive(Debug)]
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

// if name is same as field name, can use field init shorthand
fn build_user(email: String, username: String) -> User {
    User {
        email,    // field init shorthand
        username, // field init shorthand
        active: true,
        sign_in_count: 1,
    }
}
// Like JS destructuring

// Use trait for struct
#[derive(Debug, Copy, Clone)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    // same name as getter
    fn width(&self) -> u32 {
        self.width + 1
    }

    // Associated function
    fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }
}

fn area(rectangle: Rectangle) -> u32 {
    rectangle.area()
}

fn main() {
    nested_loop();
    tuple();
    array();
    let user1 = build_user(
        String::from("zhoujingya@gmail.com"),
        String::from("zhoujingya"),
    );
    let user2 = User {
        email: String::from("254644528@qq.com"),
        ..user1 // 类似于JS的解构赋值，此时user name会失效, 因为只有movinit trait
                // 没有copyinit trait
    };
    // println!(
    //     "user1: email: {}, username: {}",
    //     user1.email, user1.username
    // );
    println!(
        "user2: email: {}, username: {}",
        user2.email, user2.username
    );
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    println!("rect1 is {:#?}", rect1); // pretty print
    println!(
        "The area of the rectangle is {} square pixels.",
        area(rect1)
    );
    println!("rect1 width is {}", rect1.width());
    println!("rect1 is {:#?}", rect1);
    let rect2 = Rectangle::square(3);
    println!("rect2 is {:#?}", rect2);
}
