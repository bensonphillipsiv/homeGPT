#!/bin/bash
# install_service.sh
# Installs the HomeGPT audio streamer as a systemd service
#
# Usage: sudo ./install_service.sh

set -e

# Configuration
SERVICE_NAME="homegpt-audio-streamer"
INSTALL_DIR="/opt/homegpt-audio-streamer"
USER="${SUDO_USER:-pi}"
UV_PATH="/home/${USER}/.local/bin/uv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing HomeGPT Audio Streamer Service${NC}"
echo "============================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo: sudo ./install_service.sh${NC}"
    exit 1
fi

# Check if uv is installed
if [ ! -f "$UV_PATH" ]; then
    echo -e "${YELLOW}uv not found at $UV_PATH${NC}"
    echo "Installing uv..."
    sudo -u "$USER" curl -LsSf https://astral.sh/uv/install.sh | sudo -u "$USER" sh
fi

# Check for portaudio
if ! dpkg -l | grep -q portaudio19-dev; then
    echo "Installing portaudio19-dev..."
    apt-get update
    apt-get install -y portaudio19-dev
fi

# Create install directory
echo "Creating install directory at $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# Copy files
echo "Copying files..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/homegpt_audio_streamer.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/"

# Copy .env if it exists, otherwise copy example
if [ -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env" "$INSTALL_DIR/"
    echo -e "${GREEN}.env file copied${NC}"
else
    echo -e "${RED}Warning: No .env file found. Create one at $INSTALL_DIR/.env${NC}"
fi

# Set ownership
chown -R "$USER:$USER" "$INSTALL_DIR"

# Install dependencies with uv
echo "Installing Python dependencies..."
cd "$INSTALL_DIR"
sudo -u "$USER" "$UV_PATH" sync

# Create systemd service
echo "Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=HomeGPT Audio Streamer
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${INSTALL_DIR}
ExecStartPre=/bin/sleep 10
ExecStart=${UV_PATH} run homegpt_audio_streamer.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment
Environment="HOME=/home/${USER}"

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "Enabling service..."
systemctl enable ${SERVICE_NAME}

# Start service
echo "Starting service..."
systemctl start ${SERVICE_NAME}

# Show status
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "============================================"
echo ""
systemctl status ${SERVICE_NAME} --no-pager
echo ""
echo -e "${GREEN}Useful commands:${NC}"
echo "  View logs:      journalctl -u ${SERVICE_NAME} -f"
echo "  Restart:        sudo systemctl restart ${SERVICE_NAME}"
echo "  Stop:           sudo systemctl stop ${SERVICE_NAME}"
echo "  Disable:        sudo systemctl disable ${SERVICE_NAME}"
echo "  Edit config:    nano ${INSTALL_DIR}/.env"
echo ""

# Check if .env needs configuration
if grep -q "your_long_lived_access_token_here" "$INSTALL_DIR/.env" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Don't forget to edit ${INSTALL_DIR}/.env with your settings!${NC}"
    echo "   Then restart: sudo systemctl restart ${SERVICE_NAME}"
fi
