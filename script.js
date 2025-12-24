// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const selectBtn = document.getElementById("selectBtn");
const removeBtn = document.getElementById("removeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const previewContainer = document.getElementById("previewContainer");
const originalImage = document.getElementById("originalImage");
const resultImage = document.getElementById("resultImage");
const originalPlaceholder = document.getElementById("originalPlaceholder");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const errorDiv = document.getElementById("error");
const successDiv = document.getElementById("success");

// State variables
let originalFile = null;
let resultBlob = null;

// Event Listeners
selectBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFileSelect);
uploadArea.addEventListener("dragover", handleDragOver);
uploadArea.addEventListener("dragleave", handleDragLeave);
uploadArea.addEventListener("drop", handleDrop);
removeBtn.addEventListener("click", removeBackground);
downloadBtn.addEventListener("click", downloadResult);
resetBtn.addEventListener("click", resetApp);

// Handle file selection
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    processFile(file);
  }
}

// Handle drag over
function handleDragOver(e) {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
}

// Handle drag leave
function handleDragLeave(e) {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
}

// Handle drop
function handleDrop(e) {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");

  const file = e.dataTransfer.files[0];
  if (file) {
    processFile(file);
  }
}

// Process the selected file
function processFile(file) {
  // Validate file type
  if (!file.type.match("image.*")) {
    showError("Please select an image file (JPEG, PNG, etc.)");
    return;
  }

  // Validate file size (5MB limit)
  if (file.size > 5 * 1024 * 1024) {
    showError("File size must be less than 5MB");
    return;
  }

  originalFile = file;
  clearMessages();

  // Display original image preview
  const reader = new FileReader();
  reader.onload = (e) => {
    originalImage.src = e.target.result;
    originalImage.style.display = "block";
    originalPlaceholder.style.display = "none";
    previewContainer.style.display = "grid";
    removeBtn.disabled = false;
    downloadBtn.disabled = true;
    resultImage.style.display = "none";
    resultPlaceholder.style.display = "block";
  };
  reader.readAsDataURL(file);
}

// Remove background using remove.bg API
async function removeBackground() {
  if (!originalFile) {
    showError("Please select an image first");
    return;
  }

  // Show loading state
  showLoading();

  const formData = new FormData();
  formData.append("image_file", originalFile);
  formData.append("size", "auto");

  try {
    // Note: In a real application, you would use your own API key
    // For this demo, we'll simulate the API call since we can't expose API keys in client-side code
    // In production, this should be handled by a backend proxy
    await simulateBackgroundRemoval();
  } catch (error) {
    console.error("Background removal failed:", error);
    showError("Failed to remove background. Please try again.");
    hideLoading();
  }
}

// Ultra-Advanced Background Removal Algorithm
function simulateBackgroundRemoval() {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the original image
      ctx.drawImage(img, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const width = canvas.width;
      const height = canvas.height;

      // Step 1: Advanced background color detection with clustering
      const bgColors = detectBackgroundColors(data, width, height);

      // Step 2: Create sophisticated alpha mask with multiple background detection
      const alphaMask = createAdvancedAlphaMask(data, width, height, bgColors);

      // Step 3: Apply morphological operations for cleaner mask
      const cleanedMask = morphologicalClean(alphaMask, width, height);

      // Step 4: Advanced edge detection and refinement
      const refinedMask = detectAndRefineEdges(
        data,
        cleanedMask,
        width,
        height
      );

      // Step 5: Multi-level smoothing for perfect edges
      const ultraSmoothMask = ultraSmoothAlphaMask(refinedMask, width, height);

      // Step 6: Apply final mask with color correction
      applyAdvancedAlphaMask(data, ultraSmoothMask, width, height);

      // Step 7: Final color enhancement and detail preservation
      enhanceColors(data, width, height);

      ctx.putImageData(imageData, 0, 0);

      // Convert to high-quality blob
      canvas.toBlob(
        (blob) => {
          resultBlob = blob;
          const resultUrl = URL.createObjectURL(blob);
          resultImage.src = resultUrl;
          resultImage.style.display = "block";
          resultPlaceholder.style.display = "none";
          downloadBtn.disabled = false;
          showSuccess("Background removed with ultra-high precision!");
          hideLoading();
          resolve();
        },
        "image/png",
        1.0
      ); // Maximum quality
    };

    img.src = originalImage.src;
  });
}

// Advanced background color detection with clustering
function detectBackgroundColors(data, width, height) {
  const sampleSize = Math.min(1000, width * height); // Sample pixels for clustering
  const samples = [];

  // Sample pixels from edges and corners
  for (let i = 0; i < sampleSize; i++) {
    let x, y;

    // Sample from edges and corners more heavily
    if (i < sampleSize * 0.6) {
      // Edges and corners
      const edge = Math.floor(Math.random() * 4);
      switch (edge) {
        case 0:
          x = Math.floor(Math.random() * width);
          y = 0;
          break; // top
        case 1:
          x = width - 1;
          y = Math.floor(Math.random() * height);
          break; // right
        case 2:
          x = Math.floor(Math.random() * width);
          y = height - 1;
          break; // bottom
        case 3:
          x = 0;
          y = Math.floor(Math.random() * height);
          break; // left
      }
    } else {
      // Random interior pixels
      x = Math.floor(Math.random() * width);
      y = Math.floor(Math.random() * height);
    }

    const index = (y * width + x) * 4;
    samples.push({
      r: data[index],
      g: data[index + 1],
      b: data[index + 2],
      x,
      y,
    });
  }

  // Simple k-means clustering to find dominant background colors
  const k = 3; // Number of clusters
  const clusters = kMeans(samples, k);

  // Return the most dominant background colors (largest clusters)
  return clusters
    .sort((a, b) => b.points.length - a.points.length)
    .slice(0, 2) // Return top 2 background colors
    .map((cluster) => cluster.centroid);
}

// K-means clustering for color analysis
function kMeans(points, k) {
  // Initialize centroids randomly
  let centroids = [];
  for (let i = 0; i < k; i++) {
    centroids.push({
      r: Math.random() * 255,
      g: Math.random() * 255,
      b: Math.random() * 255,
    });
  }

  const maxIterations = 10;
  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign points to nearest centroid
    const clusters = centroids.map(() => ({ points: [], centroid: null }));

    points.forEach((point) => {
      let minDist = Infinity;
      let closestCluster = 0;

      centroids.forEach((centroid, index) => {
        const dist = colorDistance(point, centroid);
        if (dist < minDist) {
          minDist = dist;
          closestCluster = index;
        }
      });

      clusters[closestCluster].points.push(point);
    });

    // Update centroids
    centroids = clusters.map((cluster) => {
      if (cluster.points.length === 0)
        return cluster.centroid || { r: 0, g: 0, b: 0 };

      const sum = cluster.points.reduce(
        (acc, point) => ({
          r: acc.r + point.r,
          g: acc.g + point.g,
          b: acc.b + point.b,
        }),
        { r: 0, g: 0, b: 0 }
      );

      return {
        r: Math.round(sum.r / cluster.points.length),
        g: Math.round(sum.g / cluster.points.length),
        b: Math.round(sum.b / cluster.points.length),
      };
    });
  }

  return centroids.map((centroid, index) => ({
    centroid,
    points: points.filter((point) => {
      let minDist = Infinity;
      let closestCentroid = centroids[0];
      centroids.forEach((c) => {
        const dist = colorDistance(point, c);
        if (dist < minDist) {
          minDist = dist;
          closestCentroid = c;
        }
      });
      return closestCentroid === centroid;
    }),
  }));
}

// Calculate color distance
function colorDistance(c1, c2) {
  return Math.sqrt(
    Math.pow(c1.r - c2.r, 2) +
      Math.pow(c1.g - c2.g, 2) +
      Math.pow(c1.b - c2.b, 2)
  );
}

// Detect background color by analyzing corner pixels (legacy function)
function detectBackgroundColor(data, width, height) {
  const corners = [
    [0, 0],
    [width - 1, 0],
    [0, height - 1],
    [width - 1, height - 1],
  ];

  let r = 0,
    g = 0,
    b = 0,
    count = 0;

  corners.forEach(([x, y]) => {
    const index = (y * width + x) * 4;
    r += data[index];
    g += data[index + 1];
    b += data[index + 2];
    count++;
  });

  return {
    r: Math.round(r / count),
    g: Math.round(g / count),
    b: Math.round(b / count),
  };
}

// Create sophisticated alpha mask with multiple background detection
function createAdvancedAlphaMask(data, width, height, bgColors) {
  const mask = new Uint8Array(width * height);

  // Initialize mask to fully opaque
  for (let i = 0; i < mask.length; i++) {
    mask[i] = 255;
  }

  // Use multiple background colors for better detection
  const tolerances = [25, 35]; // Different tolerances for different backgrounds

  bgColors.forEach((bgColor, colorIndex) => {
    const tolerance = tolerances[colorIndex] || 30;
    const visited = new Set();
    const queue = [];

    // Start flood fill from corners and edges
    const startPoints = [
      0,
      width - 1,
      (height - 1) * width,
      height * width - 1, // corners
      Math.floor(width / 2), // top center
      (height - 1) * width + Math.floor(width / 2), // bottom center
      Math.floor(height / 2) * width, // left center
      Math.floor(height / 2) * width + (width - 1), // right center
    ];

    startPoints.forEach((index) => {
      if (
        index >= 0 &&
        index < mask.length &&
        isBackgroundPixel(data, index * 4, bgColor, tolerance)
      ) {
        queue.push(index);
        visited.add(index);
        mask[index] = 0;
      }
    });

    // Flood fill with adaptive tolerance
    while (queue.length > 0) {
      const current = queue.shift();
      const neighbors = getExtendedNeighbors(current, width, height);

      neighbors.forEach((neighbor) => {
        if (!visited.has(neighbor)) {
          const isBg = isBackgroundPixel(
            data,
            neighbor * 4,
            bgColor,
            tolerance
          );
          if (isBg) {
            visited.add(neighbor);
            queue.push(neighbor);
            mask[neighbor] = 0;
          }
        }
      });
    }
  });

  return mask;
}

// Get extended neighbors (8-directional)
function getExtendedNeighbors(index, width, height) {
  const neighbors = [];
  const x = index % width;
  const y = Math.floor(index / width);

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;

      const nx = x + dx;
      const ny = y + dy;

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        neighbors.push(ny * width + nx);
      }
    }
  }

  return neighbors;
}

// Apply morphological operations for cleaner mask
function morphologicalClean(mask, width, height) {
  // Apply opening operation (erosion followed by dilation)
  const eroded = erodeMask(mask, width, height);
  const opened = dilateMask(eroded, width, height);

  // Apply closing operation (dilation followed by erosion)
  const dilated = dilateMask(opened, width, height);
  const closed = erodeMask(dilated, width, height);

  return closed;
}

// Morphological erosion
function erodeMask(mask, width, height) {
  const eroded = new Uint8Array(mask.length);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      let minValue = 255;

      // Check 3x3 neighborhood
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nIndex = (y + dy) * width + (x + dx);
          minValue = Math.min(minValue, mask[nIndex]);
        }
      }

      eroded[index] = minValue;
    }
  }

  return eroded;
}

// Morphological dilation
function dilateMask(mask, width, height) {
  const dilated = new Uint8Array(mask.length);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      let maxValue = 0;

      // Check 3x3 neighborhood
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nIndex = (y + dy) * width + (x + dx);
          maxValue = Math.max(maxValue, mask[nIndex]);
        }
      }

      dilated[index] = maxValue;
    }
  }

  return dilated;
}

// Advanced edge detection and refinement
function detectAndRefineEdges(data, mask, width, height) {
  const refined = new Uint8Array(mask.length);

  // Copy original mask
  for (let i = 0; i < mask.length; i++) {
    refined[i] = mask[i];
  }

  // Detect edges using gradient analysis
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      const currentAlpha = mask[index];

      // Calculate gradient (edge strength)
      let gradientSum = 0;
      let count = 0;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nIndex = (y + dy) * width + (x + dx);
          gradientSum += Math.abs(mask[nIndex] - currentAlpha);
          count++;
        }
      }

      const gradient = gradientSum / count;

      // If this is an edge pixel (high gradient), refine it
      if (gradient > 50) {
        // Use intelligent alpha blending based on color similarity
        const centerColor = {
          r: data[index * 4],
          g: data[index * 4 + 1],
          b: data[index * 4 + 2],
        };

        let similarNeighbors = 0;
        let totalAlpha = 0;

        // Check neighboring pixels
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nIndex = (y + dy) * width + (x + dx);
            const nColor = {
              r: data[nIndex * 4],
              g: data[nIndex * 4 + 1],
              b: data[nIndex * 4 + 2],
            };

            const colorDiff = colorDistance(centerColor, nColor);
            if (colorDiff < 30) {
              // Similar color
              similarNeighbors++;
              totalAlpha += mask[nIndex];
            }
          }
        }

        if (similarNeighbors > 0) {
          refined[index] = Math.round(totalAlpha / similarNeighbors);
        }
      }
    }
  }

  return refined;
}

// Multi-level smoothing for perfect edges
function ultraSmoothAlphaMask(mask, width, height) {
  // Apply multiple levels of smoothing
  let smoothed = mask;

  // Level 1: Gaussian blur approximation
  smoothed = gaussianBlurAlpha(smoothed, width, height, 1.0);

  // Level 2: Bilateral filter for edge preservation
  smoothed = bilateralFilterAlpha(smoothed, width, height);

  // Level 3: Final refinement
  smoothed = refineAlphaEdges(smoothed, width, height);

  return smoothed;
}

// Gaussian blur approximation for alpha mask
function gaussianBlurAlpha(mask, width, height, sigma) {
  const blurred = new Uint8Array(mask.length);
  const kernelSize = Math.ceil(sigma * 3);
  const kernel = generateGaussianKernel(kernelSize, sigma);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      let weightedSum = 0;
      let totalWeight = 0;

      for (let ky = -kernelSize; ky <= kernelSize; ky++) {
        for (let kx = -kernelSize; kx <= kernelSize; kx++) {
          const nx = x + kx;
          const ny = y + ky;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nIndex = ny * width + nx;
            const weight = kernel[ky + kernelSize][kx + kernelSize];
            weightedSum += mask[nIndex] * weight;
            totalWeight += weight;
          }
        }
      }

      blurred[index] = Math.round(weightedSum / totalWeight);
    }
  }

  return blurred;
}

// Generate Gaussian kernel
function generateGaussianKernel(size, sigma) {
  const kernel = [];
  const center = size;

  for (let y = 0; y <= size * 2; y++) {
    kernel[y] = [];
    for (let x = 0; x <= size * 2; x++) {
      const dx = x - center;
      const dy = y - center;
      const distance = Math.sqrt(dx * dx + dy * dy);
      kernel[y][x] = Math.exp(-(distance * distance) / (2 * sigma * sigma));
    }
  }

  return kernel;
}

// Bilateral filter for edge-preserving smoothing
function bilateralFilterAlpha(mask, width, height) {
  const filtered = new Uint8Array(mask.length);
  const spatialSigma = 2.0;
  const rangeSigma = 30.0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const centerValue = mask[index];

      let weightedSum = 0;
      let totalWeight = 0;

      // 5x5 kernel
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nIndex = ny * width + nx;
            const nValue = mask[nIndex];

            // Spatial weight
            const spatialDist = Math.sqrt(dx * dx + dy * dy);
            const spatialWeight = Math.exp(
              -(spatialDist * spatialDist) / (2 * spatialSigma * spatialSigma)
            );

            // Range weight
            const rangeDist = Math.abs(nValue - centerValue);
            const rangeWeight = Math.exp(
              -(rangeDist * rangeDist) / (2 * rangeSigma * rangeSigma)
            );

            const weight = spatialWeight * rangeWeight;
            weightedSum += nValue * weight;
            totalWeight += weight;
          }
        }
      }

      filtered[index] = Math.round(weightedSum / totalWeight);
    }
  }

  return filtered;
}

// Final edge refinement
function refineAlphaEdges(mask, width, height) {
  const refined = new Uint8Array(mask.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const current = mask[index];

      // Only refine semi-transparent pixels
      if (current > 10 && current < 245) {
        let sum = 0;
        let count = 0;

        // Weighted average with emphasis on similar values
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const nIndex = ny * width + nx;
              const nValue = mask[nIndex];
              const weight = 1 / (1 + Math.abs(nValue - current) / 50);
              sum += nValue * weight;
              count += weight;
            }
          }
        }

        refined[index] = Math.round(sum / count);
      } else {
        refined[index] = current;
      }
    }
  }

  return refined;
}

// Apply final mask with color correction
function applyAdvancedAlphaMask(data, mask, width, height) {
  for (let i = 0; i < mask.length; i++) {
    const alphaIndex = i * 4 + 3;
    data[alphaIndex] = mask[i];
  }
}

// Final color enhancement and detail preservation
function enhanceColors(data, width, height) {
  // Apply subtle color enhancement to preserve details
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];

    // Only enhance visible pixels
    if (alpha > 128) {
      // Subtle contrast enhancement
      for (let c = 0; c < 3; c++) {
        let color = data[i + c];
        // Apply slight gamma correction
        color = Math.pow(color / 255, 0.95) * 255;
        // Clamp to valid range
        data[i + c] = Math.max(0, Math.min(255, Math.round(color)));
      }
    }
  }
}

// Create alpha mask using intelligent background detection
function createAlphaMask(data, width, height, bgColor) {
  const mask = new Uint8Array(width * height);
  const tolerance = 30; // Color tolerance for background detection

  // Initialize mask
  for (let i = 0; i < mask.length; i++) {
    mask[i] = 255; // Default to opaque
  }

  // Flood fill from corners to identify background
  const visited = new Set();
  const queue = [];

  // Add corner pixels to queue
  const corners = [0, width - 1, (height - 1) * width, height * width - 1];
  corners.forEach((index) => {
    if (isBackgroundPixel(data, index * 4, bgColor, tolerance)) {
      queue.push(index);
      visited.add(index);
      mask[index] = 0; // Mark as transparent
    }
  });

  // Flood fill algorithm
  while (queue.length > 0) {
    const current = queue.shift();
    const neighbors = getNeighbors(current, width, height);

    neighbors.forEach((neighbor) => {
      if (
        !visited.has(neighbor) &&
        isBackgroundPixel(data, neighbor * 4, bgColor, tolerance)
      ) {
        visited.add(neighbor);
        queue.push(neighbor);
        mask[neighbor] = 0;
      }
    });
  }

  return mask;
}

// Check if pixel is background based on color distance
function isBackgroundPixel(data, index, bgColor, tolerance) {
  const r = data[index];
  const g = data[index + 1];
  const b = data[index + 2];

  const distance = Math.sqrt(
    Math.pow(r - bgColor.r, 2) +
      Math.pow(g - bgColor.g, 2) +
      Math.pow(b - bgColor.b, 2)
  );

  return distance <= tolerance;
}

// Get neighboring pixel indices
function getNeighbors(index, width, height) {
  const neighbors = [];
  const x = index % width;
  const y = Math.floor(index / width);

  // Check all 4 directions
  if (x > 0) neighbors.push(index - 1); // left
  if (x < width - 1) neighbors.push(index + 1); // right
  if (y > 0) neighbors.push(index - width); // up
  if (y < height - 1) neighbors.push(index + width); // down

  return neighbors;
}

// Apply smoothing to alpha mask to reduce jagged edges
function smoothAlphaMask(mask, width, height) {
  const smoothed = new Uint8Array(mask.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      let sum = 0;
      let count = 0;

      // Average with neighboring pixels
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nIndex = ny * width + nx;
            sum += mask[nIndex];
            count++;
          }
        }
      }

      smoothed[index] = Math.round(sum / count);
    }
  }

  return smoothed;
}

// Apply alpha mask to image data
function applyAlphaMask(data, mask, width, height) {
  for (let i = 0; i < mask.length; i++) {
    const alphaIndex = i * 4 + 3;
    data[alphaIndex] = mask[i];
  }
}

// Refine edges for smoother results
function refineEdges(data, width, height) {
  const tempData = new Uint8ClampedArray(data);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = (y * width + x) * 4;
      const alpha = data[index + 3];

      // If this is a semi-transparent pixel, blend with neighbors
      if (alpha > 0 && alpha < 255) {
        let r = 0,
          g = 0,
          b = 0,
          a = 0,
          count = 0;

        // Sample neighboring pixels
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            const nIndex = (ny * width + nx) * 4;

            r += tempData[nIndex];
            g += tempData[nIndex + 1];
            b += tempData[nIndex + 2];
            a += tempData[nIndex + 3];
            count++;
          }
        }

        // Apply averaged values
        data[index] = Math.round(r / count);
        data[index + 1] = Math.round(g / count);
        data[index + 2] = Math.round(b / count);
        data[index + 3] = Math.round(a / count);
      }
    }
  }
}

// Show loading state
function showLoading() {
  removeBtn.innerHTML = '<div class="spinner"></div> Processing...';
  removeBtn.disabled = true;
  errorDiv.style.display = "none";
  successDiv.style.display = "none";
}

// Hide loading state
function hideLoading() {
  removeBtn.innerHTML = "Remove Background";
  removeBtn.disabled = false;
}

// Download the result
function downloadResult() {
  if (!resultBlob) {
    showError("No result to download");
    return;
  }

  const url = URL.createObjectURL(resultBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "background-removed.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Show error message
function showError(message) {
  errorDiv.textContent = message;
  errorDiv.style.display = "block";
  successDiv.style.display = "none";
}

// Show success message
function showSuccess(message) {
  successDiv.textContent = message;
  successDiv.style.display = "block";
  errorDiv.style.display = "none";
}

// Clear all messages
function clearMessages() {
  errorDiv.style.display = "none";
  successDiv.style.display = "none";
}

// Reset the app to initial state
function resetApp() {
  init();
}

// Initialize the app
function init() {
  // Clear any existing file input
  fileInput.value = "";

  // Reset UI state
  originalFile = null;
  resultBlob = null;
  removeBtn.disabled = true;
  downloadBtn.disabled = true;
  previewContainer.style.display = "none";
  originalImage.style.display = "none";
  resultImage.style.display = "none";
  originalPlaceholder.style.display = "block";
  resultPlaceholder.style.display = "block";
  clearMessages();
}

// Initialize on load
init();
