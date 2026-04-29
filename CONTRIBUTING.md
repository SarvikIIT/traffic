# Contributing to City-Scale Traffic Digital Twin

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical merits of contributions
- Help create a welcoming environment for all contributors

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check if the issue already exists in [Issues](https://github.com/SarvikIIT/traffic-digital-twin/issues)
2. Verify the bug with the latest version
3. Collect relevant information (OS, Python version, GPU details, error logs)

When submitting:
- Use a clear, descriptive title
- Provide step-by-step reproduction instructions
- Include error messages and stack traces
- Describe expected vs actual behavior

### Suggesting Features

Feature requests should:
- Have a clear use case
- Align with project goals
- Include implementation ideas if possible
- Consider performance and scalability implications

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/SarvikIIT/traffic-digital-twin.git
   cd traffic-digital-twin
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   pytest tests/
   python -m black src/
   python -m flake8 src/
   ```

5. **Commit**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

   Commit message format:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements
   - `Refactor:` for code refactoring
   - `Docs:` for documentation changes

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style
- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

Example:
```python
def calculate_traffic_density(
    vehicle_count: int,
    road_length: float
) -> float:
    """
    Calculate traffic density in vehicles per kilometer.

    Args:
        vehicle_count: Number of vehicles detected
        road_length: Length of road segment in meters

    Returns:
        Traffic density in vehicles/km
    """
    return (vehicle_count / road_length) * 1000
```

### Testing
- Write unit tests for new functions
- Maintain >80% code coverage
- Use meaningful test names
- Include edge cases

### Documentation
- Update README.md for new features
- Add docstrings to new modules
- Include code examples where helpful
- Update API documentation

## Development Setup

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v --cov=src
   ```

## Project Areas

Good areas for contributions:

- **Models**: Improve detection accuracy, add new architectures
- **Optimization**: Enhance RL algorithms, reduce inference time
- **Data**: Add support for new datasets, improve data loaders
- **Visualization**: Create better dashboards, add new metrics
- **Documentation**: Improve tutorials, add examples
- **Testing**: Increase coverage, add integration tests
- **Performance**: Optimize bottlenecks, reduce memory usage

## Questions?

- Open a [Discussion](https://github.com/SarvikIIT/traffic-digital-twin/discussions)
- Check existing documentation in [docs/](docs/)
- Reach out to maintainers

Thank you for contributing!
