mod utils;

use utils::greeting;

fn main() {
  println!("{}", greeting::get_greeting());
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_greeting() {
    assert_eq!(greeting::get_greeting(), "Hello, world!");
  }
}
