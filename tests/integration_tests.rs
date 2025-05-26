// 集成测试 - 运行所有示例的测试
use std::process::Command;

#[test]
fn test_yolov3_burn_example() {
  // 运行 YOLOv3 Burn 示例的测试
  let output = Command::new("cargo")
    .args(["test", "--example", "yolov3_burn"])
    .output()
    .expect("Failed to execute cargo test");

  if !output.status.success() {
    panic!(
      "YOLOv3 Burn example tests failed:\nstdout: {}\nstderr: {}",
      String::from_utf8_lossy(&output.stdout),
      String::from_utf8_lossy(&output.stderr)
    );
  }

  // 检查测试输出包含预期的成功信息
  let stdout = String::from_utf8_lossy(&output.stdout);
  assert!(stdout.contains("test result: ok"));
  assert!(stdout.contains("6 passed"));
}

#[test]
fn test_yolov3_burn_example_runs() {
  // 测试 YOLOv3 Burn 示例能够运行
  let output = Command::new("cargo")
    .args(["run", "--example", "yolov3_burn"])
    .output()
    .expect("Failed to execute cargo run");

  if !output.status.success() {
    panic!(
      "YOLOv3 Burn example failed to run:\nstdout: {}\nstderr: {}",
      String::from_utf8_lossy(&output.stdout),
      String::from_utf8_lossy(&output.stderr)
    );
  }

  // 检查输出包含预期的信息
  let stdout = String::from_utf8_lossy(&output.stdout);
  assert!(stdout.contains("YOLOv3 Burn Example"));
  assert!(stdout.contains("YOLOv3 Burn Example Finished"));
}

#[test]
fn test_yolov3_burn_clippy() {
  // 测试 YOLOv3 Burn 示例通过 Clippy 检查
  let output = Command::new("cargo")
    .args(["clippy", "--example", "yolov3_burn", "--", "-D", "warnings"])
    .output()
    .expect("Failed to execute cargo clippy");

  if !output.status.success() {
    panic!(
      "YOLOv3 Burn example failed clippy check:\nstdout: {}\nstderr: {}",
      String::from_utf8_lossy(&output.stdout),
      String::from_utf8_lossy(&output.stderr)
    );
  }
}
