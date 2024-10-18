enum Message {
    Quit,
    Move(i32, i32),
    Write(String),
    ChangeColor(i32, i32, i32, i32),
}

fn main() {
    let a = Message::Move(1, 2);
    let b = Message::Write(String::from("hello"));
    let c = Message::ChangeColor(1, 2, 3, 4);
    fn print_message(msg: &Message) {
        match msg {
            Message::Move(x, y) => println!("Move to ({}, {})", x, y),
            Message::Write(s) => println!("Write: {}", s),
            Message::ChangeColor(r, g, b, a) => {
                println!("ChangeColor: ({}, {}, {}, {})", r, g, b, a)
            }
            Message::Quit => println!("Quit"),
        }
    }

    print_message(&a);
    print_message(&b);
    print_message(&c);
    // if let to match a single variant
    if let Message::Move(x, y) = a {
        println!("Move to ({}, {})", x, y);
    }
}
