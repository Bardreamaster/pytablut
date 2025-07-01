#!/bin/bash

# Exit on error
set -e

# Function to display usage instructions
display_usage() {
    echo "Usage: $0 --name PACKAGE_NAME --author AUTHOR_NAME --email AUTHOR_EMAIL"
    echo
    echo "Configure a Python package with custom name, author, and email."
    echo "Required arguments:"
    echo "  --name    Package name (use-hyphens-for-multi-word-names)"
    echo "  --author  Author's name"
    echo "  --email   Author's email"
    echo
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --name)
        PACKAGE_NAME="$2"
        shift 2
        ;;
        --author)
        AUTHOR_NAME="$2"
        shift 2
        ;;
        --email)
        AUTHOR_EMAIL="$2"
        shift 2
        ;;
        --help|-h)
        display_usage
        ;;
        *)
        echo "Unknown option: $1"
        display_usage
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$PACKAGE_NAME" ] || [ -z "$AUTHOR_NAME" ] || [ -z "$AUTHOR_EMAIL" ]; then
    echo "Error: missing required arguments"
    display_usage
fi

# Convert hyphenated package name to underscore format for Python directory
PACKAGE_NAME_UNDERSCORE=$(echo $PACKAGE_NAME | tr '-' '_')

# Get current year
CURRENT_YEAR=$(date +%Y)

echo "Initializing Python package:"
echo "Package Name: $PACKAGE_NAME"
echo "Python Module Name: $PACKAGE_NAME_UNDERSCORE"
echo "Author: $AUTHOR_NAME"
echo "Email: $AUTHOR_EMAIL"

# Update pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/name = \"python-package-template\"/name = \"$PACKAGE_NAME\"/" pyproject.toml
sed -i "s/{ name = \"changshan\", email = \"changshanshi@outlook.com\" }/{ name = \"$AUTHOR_NAME\", email = \"$AUTHOR_EMAIL\" }/" pyproject.toml

# Update the source directory name
echo "Renaming source directory..."
if [ -d "src/python_package_template" ]; then
    mkdir -p "src/$PACKAGE_NAME_UNDERSCORE"
    # Copy files from old directory to new directory
    cp -r src/python_package_template/* "src/$PACKAGE_NAME_UNDERSCORE/"
    # Remove old directory
    rm -rf src/python_package_template
    echo "Source directory renamed from 'src/python_package_template' to 'src/$PACKAGE_NAME_UNDERSCORE'"
else
    echo "Warning: src/python_package_template directory not found"
fi

# Update README.md
echo "Updating README.md..."
cat > README.md << EOF
# $PACKAGE_NAME

## Installation

\`\`\`bash
python3 -m pip install $PACKAGE_NAME
\`\`\`

## Usage

\`\`\`python
import $PACKAGE_NAME_UNDERSCORE

# Add usage examples here
\`\`\`

## License

MIT License
EOF

# Update documentation info
echo "Updating documentation..."
sed -i "s/project = \"python_package_template\"/project = \"$PACKAGE_NAME_UNDERSCORE\"/" docs/source/conf.py
sed -i "s/copyright = \"2025, Wei Jinqi\"/copyright = \"$CURRENT_YEAR, $AUTHOR_NAME\"/" docs/source/conf.py
sed -i "s/author = \"Wei Jinqi\"/author = \"$AUTHOR_NAME\"/" docs/source/conf.py
sed -i "s/\"..\/..\/src\/python_package_template\"/\"..\/..\/src\/$PACKAGE_NAME_UNDERSCORE\"/" docs/source/conf.py
sed -i "s/python_package_template/$PACKAGE_NAME_UNDERSCORE/" docs/source/installation.md

cd ..
# Get the current directory name
CURRENT_DIR=$(basename "$PWD")
CURRENT_DIR="${CURRENT_DIR%/}"
# Check if the current directory is the same as the package name
if [ "$CURRENT_DIR" != "$PACKAGE_NAME_UNDERSCORE" ]; then
    echo "Renaming current directory to match package name..."
    cd ..
    mv "$CURRENT_DIR" "$PACKAGE_NAME_UNDERSCORE"
    cd "$PACKAGE_NAME_UNDERSCORE"
fi

echo "Package successfully initialized!"
echo "Next steps:"
echo "1. Add your code to src/$PACKAGE_NAME_UNDERSCORE"
echo "2. Update the documentation in docs/source/"
echo "3. Build your package with: python -m build"
echo "4. Build documentation with: cd docs && make html"
