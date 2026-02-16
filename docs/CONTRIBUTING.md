# GS-ICP-SLAM ROS2 Package - Contributing Guide

This document provides guidelines and instructions for contributing to the 4D Gaussian Splatting SLAM ROS2 package.

## Getting Started

### Before You Begin

1. **Read the Documentation**: Thoroughly read the main [README](README.md), [Developer Guide](DEVELOPMENT.md), and [API Reference](API.md) to understand the project structure and code philosophy.

2. **Set Up Your Environment**: Follow the development setup instructions in the [DEVELOPMENT.md](DEVELOPMENT.md#development-environment).

3. **Build and Test**: Ensure you can successfully build and run the package before making changes.

### Communication

Before starting significant work, it's recommended to:
- Open an issue to discuss your proposed changes
- Join the project discussions on GitHub
- Ask questions in the appropriate channels

## Code of Conduct

Please be respectful and constructive in all interactions. This project aims to be an inclusive, helpful community for SLAM and robotics research.

## Coding Standards

### Python Style Guide

- **Formatting**: Use `black` for automatic code formatting with line length 100
- **Linting**: Use `flake8` for code quality checks
- **Type Checking**: Use `mypy` for static type checking
- **Testing**: Use `pytest` for unit and integration tests

```bash
# Install style checkers
pip install black flake8 mypy pytest

# Run formatting
black src/ tests/ examples/

# Run linting
flake8 src/ tests/ examples/

# Run type checking
mypy src/

# Run tests
pytest tests/
```

### ROS2 Specific Guidelines

- Use standard ROS2 node structure and patterns
- Follow ROS2 naming conventions for topics, services, and parameters
- Implement proper ROS2 logging
- Use QoS profiles appropriately
- Handle exceptions gracefully

## Project Structure

```
gs_icp_slam_ros2/
├── src/gs_icp_slam_ros2/          # Python source code
├── tests/                       # Test files
├── examples/                    # Usage examples
├── config/                      # Configuration files
├── launch/                      # ROS2 launch files
├── scripts/                     # Utility scripts
├── docs/                        # Documentation
└── CMakeLists.txt               # Build configuration
```

## Contribution Areas

### Feature Development

1. **SLAM Core Algorithms**: Implement or improve SLAM algorithms
2. **GPU Optimization**: Add CUDA implementations for performance
3. **Visualization**: Enhance rendering and visualization tools
4. **Integration**: Add support for new sensors or platforms
5. **Testing**: Improve test coverage and add new tests

### Bug Fixes

1. Identify and report bugs through GitHub Issues
2. Provide minimal reproduction cases
3. Submit PRs with clear bug descriptions
4. Ensure existing tests still pass

### Documentation

1. Improve existing documentation
2. Add missing API documentation
3. Create tutorials for new features
4. Update configuration examples
5. Add code comments where necessary

### Testing

1. Write unit tests for new functions
2. Add integration tests for complex workflows
3. Ensure all tests pass before submitting
4. Maintain test coverage >80%

## Development Workflow

### 1. Fork and Branch

```bash
# Fork the repository on GitHub

# Clone your fork locally
git clone https://github.com/yourusername/gs_icp_slam_ros2.git
cd gs_icp_slam_ros2

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Write your code, follow the coding standards, and document your changes thoroughly.

### 3. Test Your Changes

```bash
# Build the package
colcon build --packages-select gs_icp_slam --symlink-install

# Run all tests
colcon test --packages-select gs_icp_slam
colcon test-result --verbose

# Run specific tests
pytest tests/test_parameters.py -v
```

### 4. Commit Your Changes

Follow conventional commit messages:

```bash
git add .
git commit -m "feat: add new feature with detailed description"
```

Conventional commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test-related changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

### 5. Push and Open a Pull Request

```bash
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
```

### 6. Address Review Feedback

- Respond to review comments promptly
- Make requested changes
- Squash commits if needed
- Continue to improve based on feedback

## Pull Request Template

```markdown
## Description
Brief description of the PR and its purpose.

## Changes
- Change 1: Description
- Change 2: Description
- etc.

## Testing
- [ ] Tests added/updated for new functionality
- [ ] All existing tests pass
- [ ] Test coverage maintained or improved
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated (README, API, etc.)
- [ ] No merge conflicts
- [ ] Commit messages clear and descriptive
- [ ] PR target branch is main
- [ ] Description explains the "why" and "how"
- [ ] Screenshot/GIF if applicable
```

## Commit Message Guidelines

Follow these conventions for clear, organized commit history:

```plaintext
feat: add real-time SLAM tracking

This commit adds the core SLAM tracking functionality including:
- Feature extraction and matching
- Pose estimation using optical flow
- Keyframe selection logic
- Multi-threaded processing support

Resolves #123
```

**Good commit message structure:**
1. Type: `feat/fix/docs/test/...`
2. Scope: Optional, what area of code
3. Description: Clear, concise statement
4. Details: Bullet points for complex changes (optional)
5. Issue reference: `Fixes #123`

## Code Review Process

### What to Expect

- Reviewers will check:
  - Code quality and style
  - Correctness and efficiency
  - Test coverage
  - Documentation completeness
  - Integration with existing codebase

### Common Feedback

- **Style issues**: Use formatters and linters
- **Missing tests**: Add appropriate test cases
- **Documentation gaps**: Update docs or add comments
- **Design concerns**: Discuss implementation alternatives
- **Performance**: Optimize critical paths

### How to Respond

- Acknowledge review feedback
- Ask clarifying questions if needed
- Make requested changes
- Rebase if necessary
- Continue working on other items

## Testing Requirements

### Unit Testing

```python
# tests/test_slam_core.py

import unittest
from unittest.mock import Mock, patch
from gs_icp_slam_ros2.node import SLAMNode

class TestSLAMCore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.slam_node = SLAMNode()
    
    def test_initialization(self):
        """Test node initialization"""
        self.assertIsNotNone(self.slam_node)
    
    def test_frame_processing(self):
        """Test frame processing with valid input"""
        # Implementation
```

### Integration Testing

```bash
# Test end-to-end workflows
colcon test --packages-select gs_icp_slam --event-handlers console_direct+
```

## Documentation Requirements

### API Documentation

Every public class and method must have:

```python
class MyClass:
    """
    Brief class description.
    
    Extended description of the class purpose, usage, and important
    functionality. Should cover when and why to use the class.
    """
    
    def method(self, param: type) -> return_type:
        """
        Brief method description.
        
        Extended description of the method behavior.
        
        Args:
            param: Parameter description
            
        Returns:
            Return value description
            
        Raises:
            ExceptionType: Exception description
            
        Example:
            >>> result = obj.method(value)
            >>> print(result)
    """
```

### README Updates

- Update installation instructions if needed
- Add quick start guide for new features
- Document new parameters and configurations
- Update examples if they changed

### Tutorials

- Create step-by-step guides for complex features
- Include screen shots or code snippets
- Add "Common Pitfalls" sections
- Provide troubleshooting tips

## Performance Considerations

When contributing:

1. **Profile before optimizing**: Identify actual bottlenecks
2. **Benchmark results**: Maintain or improve performance metrics
3. **Write benchmarks**: Add test-benchmarks for performance regressions
4. **Profile code**: Use cProfile and timeit for analysis

```python
# Example benchmark
import timeit

def benchmark_function(func):
    execution_time = timeit.timeit(func, number=10000)
    print(f"{func.__name__} executed in {execution_time:.4f} seconds")
```

## Code Review Tips

### Before Submitting

- [ ] Code passes all linters
- [ ] Code passes all tests
- [ ] Code follows style guides
- [ ] Documentation is complete
- [ ] Commit messages are clear
- [ ] PR description explains changes

### What Reviewers Look For

- Correctness and robustness
- Code efficiency and minimal complexity
- Appropriate error handling
- Clear naming and structure
- Complete documentation
- Good test coverage

## Troubleshooting Common Issues

### Build Errors

- Ensure all dependencies are installed
- Check ROS2 environment is sourced
- Verify CMake configuration
- Look for typos or syntax errors

### Test Failures

- Check if environment differs from test
- Review error messages carefully
- Ensure correct ROS2 version compatibility
- Verify test dependencies are installed

### Documentation Issues

- Check formatting and structure
- Verify links work correctly
- Test examples if provided
- Update examples with new functionality

## Questions? Feedback?

- Open an issue on GitHub
- Join project discussions
- Reach out to maintainers
- Contribute improvements

## Recognition

All contributors will be:

1. Listed in the project's contributors file
2. Mentioned in the README if significantly contributing
3. Cited in papers using this package (if applicable)
4. Recognized for their code quality and spirit of collaboration

## Continuous Improvement

We are committed to maintaining high standards for this package. While we welcome all contributions, we may need to make adjustments, provide guidance, or ask for modifications to maintain code quality and consistency.

## License

Contributions to this project are accepted under the MIT License. Please ensure your contributions are your original work or properly licensed.

## Thank You!

Your contributions help improve SLAM research and applications. We appreciate your effort and expertise!

---

For more information, refer to:
- [Main README](README.md)
- [Developer Guide](DEVELOPMENT.md)
- [API Reference](API.md)
- [Issues](https://github.com/yourusername/gs_icp_slam_ros2/issues)