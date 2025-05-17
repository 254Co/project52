# Contributing to Chen3

Thank you for your interest in contributing to Chen3! This document provides guidelines and instructions for contributing to the project.

## Code Style

We follow PEP 8 style guidelines for Python code. Please ensure your code adheres to these standards:

- Use 4 spaces for indentation
- Maximum line length of 88 characters
- Use descriptive variable names
- Add docstrings for all public functions and classes
- Include type hints for function parameters and return values

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/chen3.git
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Testing

We use pytest for testing. Please ensure all new code includes appropriate tests:

1. Write tests in the `tests/` directory
2. Run tests with:
   ```bash
   pytest
   ```
3. Ensure all tests pass before submitting a pull request

## Documentation

1. Update docstrings for any new or modified functions/classes
2. Update README.md if necessary
3. Add examples in the `examples/` directory for new features

## Pull Request Process

1. Create a new branch for your feature/fix
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation
5. Submit a pull request with a clear description of changes

## Code Review

All submissions require review. We use GitHub pull requests for this purpose. Please ensure:

1. Your code is well-documented
2. Tests are included and passing
3. The PR description clearly explains the changes
4. You've addressed any feedback from reviewers

## License

By contributing to Chen3, you agree that your contributions will be licensed under the project's Apache License 2.0.
