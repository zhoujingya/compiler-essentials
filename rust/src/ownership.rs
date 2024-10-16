fn main() {
  let str: String = String::from("hello");
  // String only has move init
  let a: String = str;
  // promote
  println!("String: {}", a);
  // println!("String: {}", str); // will get error

  // Int has copy init
  let x: i32 = 5;
  let y: i32 = x;
  println!("Int: {}", x);
  println!("Int: {}", y);

  // Copy trait
  let s1: String = String::from("hello");
  let s2: String = String::from(" world");
  let s3: String = s1 + &s2;
  println!("String: {}", s3);

}
