# RAG System Testing Framework

This directory contains the testing framework for the RAG system. The framework provides comprehensive testing capabilities including retrieval accuracy, answer quality, and performance metrics.

## Setup

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Set up test data:
```bash
python test_data_setup.py
```

## Test Components

The testing framework consists of several components:

1. **Test Data Management** (`test_data_setup.py`):
   - Manages test documents and queries
   - Creates isolated test environment
   - Handles test data versioning

2. **RAG System Tests** (`test_rag_system.py`):
   - Retrieval performance testing
   - Answer quality evaluation
   - System performance metrics
   - Automated test generation

3. **Test Runner** (`run_tests.py`):
   - Executes test suite
   - Generates coverage reports
   - Creates HTML test reports
   - Supports parallel execution

## Running Tests

1. **Run all tests**:
```bash
python run_tests.py
```

2. **Run specific test categories**:
```bash
# Run only retrieval tests
pytest test_rag_system.py -k "retrieval"

# Run only quality tests
pytest test_rag_system.py -k "quality"

# Run only performance tests
pytest test_rag_system.py -k "performance"
```

## Test Reports

After running tests, you'll find the following reports in the `test_reports` directory:

1. **Coverage Report**: 
   - HTML report showing code coverage
   - Located at `test_reports/coverage_report/index.html`

2. **Test Results**:
   - Detailed test execution results
   - Located at `test_reports/test_report_[timestamp].html`

3. **Test Output**:
   - Raw test output and logs
   - Located at `test_reports/test_output_[timestamp].txt`

## Adding Custom Tests

1. **Add Test Queries**:
   Edit `test_data_setup.py` to add domain-specific test queries:
   ```python
   test_queries = [
       {
           "query": "Your test question?",
           "expected_sources": ["relevant_doc.pdf"],
           "type": "your_category"
       }
   ]
   ```

2. **Add Test Cases**:
   Add new test functions to `test_rag_system.py`:
   ```python
   def test_your_feature(rag_evaluator):
       # Your test implementation
       pass
   ```

## Performance Thresholds

Current performance thresholds:

1. **Retrieval Performance**:
   - Precision: ≥ 0.6
   - Recall: ≥ 0.6
   - F1 Score: ≥ 0.6
   - Retrieval Time: ≤ 2.0s

2. **Answer Quality**:
   - ROUGE-1: ≥ 0.4
   - ROUGE-2: ≥ 0.2
   - ROUGE-L: ≥ 0.3
   - BLEU Score: ≥ 0.2

3. **System Performance**:
   - Average Retrieval Time: ≤ 2.0s
   - Average Response Time: ≤ 5.0s
   - 95th Percentile Retrieval Time: ≤ 3.0s
   - 95th Percentile Response Time: ≤ 7.0s

## Troubleshooting

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Test Data Issues**:
   ```bash
   python test_data_setup.py --clean
   python test_data_setup.py
   ```

3. **Performance Issues**:
   - Check system resources
   - Reduce parallel test execution
   - Adjust performance thresholds if needed 