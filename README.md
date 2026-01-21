# ML Lifecycle Case Study: AI-Powered Music Recommendation Bot

A production-style MLOps project demonstrating the full machine learning lifecycle, from model integration to multi-cloud deployment with observability.

## Project Overview

This project showcases **ML Development and Operations (MLOps)** best practices through a music recommendation chatbot that combines:

- **HuggingFace LLMs** (Zephyr-7B-beta) for conversational AI
- **Spotify API** integration for personalized music recommendations
- **Multi-cloud deployment** across AWS, Azure, and GCP
- **CI/CD pipelines** with GitHub Actions
- **Observability** with Prometheus metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                      (Gradio Web App)                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Application Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Zephyr-7B LLM  │  │   Spotify API   │  │   Prometheus    │  │
│  │  (HuggingFace)  │  │  (Spotipy SDK)  │  │    Metrics      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Infrastructure Layer                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│  │    AWS    │  │   Azure   │  │    GCP    │  │  HuggingFace │  │
│  │           │  │ Container │  │           │  │    Spaces    │  │
│  │           │  │   Apps    │  │           │  │              │  │
│  └───────────┘  └───────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technology |
|----------|------------|
| **ML/AI** | HuggingFace Inference API, Zephyr-7B-beta LLM |
| **External APIs** | Spotify Web API (via Spotipy) |
| **Frontend** | Gradio |
| **Containerization** | Docker |
| **Cloud Platforms** | AWS, Azure Container Apps, GCP, HuggingFace Spaces |
| **CI/CD** | GitHub Actions |
| **Observability** | Prometheus, Grafana, Node Exporter |
| **Language** | Python 3.10+ |

## Features

### Music Recommendations
- Enter any song and artist to get personalized Spotify recommendations
- AI-enhanced insights about recommended tracks using Zephyr-7B
- Streaming responses for real-time interaction

### Observability Metrics
The application exposes Prometheus metrics on port 8000:
- `app_recommendations_processed` - Total recommendation requests
- `app_successful_recommendations` - Successful API calls
- `app_failed_recommendations` - Failed API calls
- `app_recommendation_duration_seconds` - Request latency
- `app_user_interactions` - Total user interactions
- `app_cancelled_recommendations` - Cancelled requests

## Project Structure

```
ML-Lifecycle-Case-Study/
├── app.py                    # Main application with Spotify + LLM integration
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Full observability stack (App + Prometheus + Grafana)
├── requirements.txt          # Python dependencies
├── prometheus.yml            # Prometheus scrape configuration
├── test_model.py             # Unit tests for Spotify recommendations
├── .github/
│   └── workflows/
│       ├── main.yml          # CI/CD: Auto-deploy to HuggingFace Spaces
│       ├── unit_tests.yml    # CI: Automated testing
│       └── check.yml         # CI: File size validation
├── grafana/
│   ├── dashboards/
│   │   └── music-recommendation-bot.json  # Pre-built Grafana dashboard
│   └── provisioning/
│       ├── dashboards/dashboards.yml      # Dashboard auto-provisioning
│       └── datasources/datasources.yml    # Prometheus datasource config
├── azure_build_and_start.sh  # Azure Container Apps deployment
├── azure_clean.sh            # Azure resource cleanup
├── deploy_first_part.sh      # Remote server setup script
├── deploy_second_part.sh     # Remote server deployment script
└── Prometheus/
    ├── Dockerfile            # Prometheus container
    └── prometheus.yml        # Prometheus configuration
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Spotify Developer Account ([Create one here](https://developer.spotify.com/dashboard))

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/adequatej/CS553_casestudy1.git
   cd CS553_casestudy1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   - Web UI: http://localhost:7860
   - Prometheus metrics: http://localhost:8000

### Docker Deployment

```bash
# Build the image
docker build -t music-recommendation-bot .

# Run the container
docker run -p 7860:7860 -p 8000:8000 -p 9100:9100 music-recommendation-bot
```

### Full Observability Stack (Docker Compose)

Run the complete stack with Prometheus and Grafana:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Access Points:**
- 🎵 **Application**: http://localhost:7860
- 📊 **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- 📈 **Prometheus**: http://localhost:9090

The Grafana dashboard is auto-provisioned with panels for:
- Total/Successful/Failed recommendations
- Request rate over time
- Latency metrics (avg and P95)
- Success rate gauge
- Request breakdown pie chart

### Azure Container Apps Deployment

```bash
# Build and deploy to Azure
./azure_build_and_start.sh

# Clean up resources
./azure_clean.sh
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. **On Push to Main:**
   - Runs unit tests (`unit_tests.yml`)
   - Checks file sizes for HuggingFace compatibility (`check.yml`)
   - Auto-deploys to HuggingFace Spaces (`main.yml`)

2. **On Pull Request:**
   - Validates file sizes to prevent large file commits

## Running Tests

```bash
pytest test_model.py -v
```

Tests use mocking to avoid requiring real Spotify API credentials.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token (for CI/CD deployment) |

**Note:** Spotify credentials are entered through the UI at runtime for security.

## Deployment History

This project was deployed across multiple cloud platforms as part of a graduate MLOps course:
- **HuggingFace Spaces** - Primary deployment (auto-deployed via GitHub Actions)
- **Azure Container Apps** - Containerized deployment with scripts included
- **AWS & GCP** - Deployed during course (resources since decommissioned)

## License

This project was created for educational purposes as part of the Graduate CS553 Machine Learning course.
