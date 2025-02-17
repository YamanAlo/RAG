import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_tests():
    """Run the test suite and generate reports."""
    # Create reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for report files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run tests with coverage
    print("Running tests with coverage...")
    result = subprocess.run([
        "pytest",
        "test_rag_system.py",
        "-v",
        "--cov=../",
        "--cov-report=html:./test_reports/coverage_report",
        "-n", "auto",  # Parallel execution
        f"--html=./test_reports/test_report_{timestamp}.html",
        "--self-contained-html"
    ], capture_output=True, text=True)
    
    # Save test output
    output_file = reports_dir / f"test_output_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nErrors/Warnings:\n")
            f.write(result.stderr)
            
    # Print summary
    print("\nTest Execution Summary:")
    print(f"Test output saved to: {output_file}")
    print(f"Coverage report: {reports_dir}/coverage_report/index.html")
    print(f"HTML report: {reports_dir}/test_report_{timestamp}.html")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    exit(exit_code) 