use rand::Rng;
use std::cmp::Ordering;
use std::io;
fn main() {
    println!("Guess the number!");
    let secret_number = generate_secret_number();
    loop {
        println!("Please input your guess.");

        let mut guess = String::new();
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");
        // Change variable type from String to u32
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please type a number!");
                continue;
            }
        };
        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}

fn generate_secret_number() -> u32 {
    rand::thread_rng().gen_range(1..=100)
}
