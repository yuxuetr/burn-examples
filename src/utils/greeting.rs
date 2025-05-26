/// 返回一个问候语
pub fn get_greeting() -> String {
  "Hello, world!".to_string()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_get_greeting() {
    assert_eq!(get_greeting(), "Hello, world!");
  }
}
