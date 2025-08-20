# Agentic AI Video CPE Test Platform

An AI-powered testing platform for Set-Top Boxes (STB) and Smart TVs that uses vision-language models to understand screen content and automatically execute test scenarios.

## Features

- **AI Vision Analysis**: Uses LLaVA model to understand TV screen content
- **Automated Testing**: Execute BDD test scenarios written in natural language
- **Video Capture**: Real-time HDMI stream processing
- **Device Control**: Automated IR key presses and power control
- **Docker Deployment**: Fully containerized solution

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  HDMI Capture   │────▶│ Video Module │────▶│  AI Agent   │
│     Card        │     │   (OpenCV)   │     │  (LLaVA)    │
└─────────────────┘     └──────────────┘     └─────────────┘
                                                     │
                                                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Test Report   │◀────│ BDD Runner   │◀────│   Decision  │
│   Generator     │     │ (pytest-bdd) │     │   Engine    │
└─────────────────┘     └──────────────┘     └─────────────┘
                                                     │
                                                     ▼
                        ┌──────────────┐     ┌─────────────┐
                        │ Device APIs  │◀────│ IR Control  │
                        │              │     │   Module    │
                        └──────────────┘     └─────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- HDMI capture card with RTSP/HTTP stream
- IR blaster with API support
- Python 3.11+ (for local development)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic-ai-video-test-platform
```

2. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start with Docker Compose:
```bash
docker-compose up -d
```

### Running Tests

Create a test scenario file `tests/features/netflix_launch.feature`:

```gherkin
Feature: Netflix App Launch
  Scenario: Launch Netflix from App Rail
    Given the device is on home screen
    When I navigate to Netflix in app rail
    And I press OK
    Then Netflix should launch
    And I should see either login screen or profile selection or home screen
    And I should not see black screen
```

Run the test:
```bash
docker-compose exec runner pytest tests/features/netflix_launch.feature
```

## Development

### Local Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run services:
```bash
# Start Ollama
ollama serve

# Pull LLaVA model
ollama pull llava:7b

# Start the platform
python -m src.main
```

### Project Structure

```
├── src/
│   ├── capture/      # Video capture and frame extraction
│   ├── agent/        # AI vision agent and decision engine
│   ├── control/      # Device control (IR, power)
│   ├── runner/       # BDD test runner
│   ├── api/          # FastAPI endpoints
│   └── utils/        # Utilities and helpers
├── tests/
│   └── features/     # BDD test scenarios
├── docker/           # Docker configurations
├── config/           # Configuration files
└── reports/          # Test reports output
```

## Configuration

### Video Capture
- Supports RTSP and HTTP streams
- Configurable FPS (default: 5)
- Automatic screenshot capture

### AI Model
- LLaVA 7B via Ollama
- CPU-optimized for edge deployment
- Configurable timeout and retries

### Device Control
- RESTful API integration
- Support for custom IR codes
- Power management capabilities

## API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.