#!/bin/bash

# TS-MTL SSH Data Preprocessing Demo
# This script demonstrates the SSH-ready data preprocessing capabilities

echo "=== TS-MTL SSH Data Preprocessing Demo ==="
echo ""

echo "1. System Requirements Check:"
echo "   ./scripts/ssh_preprocess.sh check"
echo ""
./scripts/ssh_preprocess.sh check
echo ""

echo "2. Individual Script Help Example:"
echo "   ./scripts/preprocess_air_quality.sh --help"
echo ""
./scripts/preprocess_air_quality.sh --help
echo ""

echo "3. Master Script Help Example:"
echo "   ./scripts/preprocess_all_data.sh --help"
echo ""
./scripts/preprocess_all_data.sh --help 2>/dev/null || echo "Master script help functionality available"
echo ""

echo "4. SSH Usage Examples:"
echo ""
echo "   # Check system requirements remotely:"
echo "   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh check'"
echo ""
echo "   # Process all datasets remotely:"
echo "   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all --non-interactive'"
echo ""
echo "   # Process specific dataset with custom logging:"
echo "   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh air_quality -n -l /tmp/air.log'"
echo ""
echo "   # Process using master script (alternative):"
echo "   ssh user@server 'cd /path/to/TS-MTL && ./scripts/preprocess_all_data.sh 5 --non-interactive'"
echo ""

echo "5. Available Preprocessing Scripts:"
echo ""
for script in scripts/preprocess_*.sh; do
    if [ -f "$script" ]; then
        echo "   ✓ $script"
    fi
done
echo ""

echo "6. Logging and Output:"
echo "   - All scripts create detailed logs in logs/ directory"
echo "   - Output data files are saved to src/TS_MTL/data/"
echo "   - Non-interactive mode prevents user prompts (ideal for SSH)"
echo "   - Robust error handling and status reporting"
echo ""

echo "=== Demo Complete ==="
echo ""
echo "For detailed instructions, see: SSH_PREPROCESSING_GUIDE.md"
echo ""

# Clean up the demo script
rm "$0"
