#!/bin/bash
echo "=================================="
echo "Frontend Implementation Verification"
echo "=================================="
echo ""

echo "Checking directory structure..."
if [ -d "frontend/src/components" ] && [ -d "frontend/src/pages" ] && [ -d "frontend/src/store" ]; then
    echo "✓ Directory structure correct"
else
    echo "✗ Directory structure incomplete"
fi

echo ""
echo "Checking components..."
components=("AudioUpload.jsx" "ProgressBar.jsx" "WaveformPlayer.jsx" "QualityReport.jsx")
for comp in "${components[@]}"; do
    if [ -f "frontend/src/components/$comp" ]; then
        echo "✓ $comp exists"
    else
        echo "✗ $comp missing"
    fi
done

echo ""
echo "Checking pages..."
pages=("TrainingFlow.jsx" "ConversionFlow.jsx")
for page in "${pages[@]}"; do
    if [ -f "frontend/src/pages/$page" ]; then
        echo "✓ $page exists"
    else
        echo "✗ $page missing"
    fi
done

echo ""
echo "Checking store..."
if [ -f "frontend/src/store/appStore.js" ]; then
    echo "✓ appStore.js exists"
else
    echo "✗ appStore.js missing"
fi

echo ""
echo "Checking configuration..."
configs=("tailwind.config.js" "postcss.config.js" "package.json")
for config in "${configs[@]}"; do
    if [ -f "frontend/$config" ]; then
        echo "✓ $config exists"
    else
        echo "✗ $config missing"
    fi
done

echo ""
echo "Checking documentation..."
docs=("README.md" "../QUICKSTART_FRONTEND.md" "../PHASE_3_FRONTEND_COMPLETE.md")
for doc in "${docs[@]}"; do
    if [ -f "frontend/$doc" ]; then
        echo "✓ $doc exists"
    else
        echo "✗ $doc missing"
    fi
done

echo ""
echo "=================================="
echo "Verification Complete"
echo "=================================="
