#!/usr/bin/env python3
import subprocess
import sys

def run_qemu_riscv64(a_out_path):
    """Run qemu-riscv64 and return output and return code"""
    try:
        result = subprocess.run(['/opt/riscv/bin/qemu-riscv64', a_out_path],
                                capture_output=True,
                                text=True)
        return result.stdout, result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running qemu-riscv64: {e}")
        sys.exit(1)

def compare_output(expected_return_code, return_code):
    """Compare actual output with expected output and display return code"""
    if str(return_code) == str(expected_return_code):
        print("Test passed: Output matches expected result")
    else:
        print("Test failed: Output does not match expected result")
        print(f"Expected return code: {expected_return_code}")
        print(f"Actual return code: {return_code}")
        sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print("Usage: python test.py <a.out_path> <expected_return_code>")
        sys.exit(1)

    a_out_path = sys.argv[1]
    expected_return_code = sys.argv[2]
    # Run qemu-riscv64 and get output and return code
    actual_output, return_code = run_qemu_riscv64(a_out_path)

    # Set expected standard output here
    # Compare output and display return code
    compare_output(expected_return_code, return_code)

if __name__ == "__main__":
    main()
