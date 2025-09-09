#!/usr/bin/env python3
"""
SIMPLE POSTER TEST - Basic functionality check
"""

import sys
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except Exception as e:
    print(f"❌ pandas error: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except Exception as e:
    print(f"❌ numpy error: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
except Exception as e:
    print(f"❌ matplotlib error: {e}")

try:
    # Simple plot test
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
    plt.title('Simple Test Plot', fontsize=16, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
    print("✅ Test plot saved successfully as 'test_plot.png'")
    plt.close()

except Exception as e:
    print(f"❌ Plot creation error: {e}")
    import traceback
    traceback.print_exc()

print("🎯 Basic functionality test completed")
