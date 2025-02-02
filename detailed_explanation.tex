WHAT IS INTERPOLATION? 🤔

Interpolation is a method used to estimate unknown values between known values.

In image processing, it helps fill in missing pixel values when resizing images.

If we enlarge an image, we need to create new pixels.

If we shrink an image, we need to remove pixels while preserving details.

Without interpolation, resizing can make images look pixelated or blurry! 😵

1️⃣ Nearest-Neighbor Interpolation (cv2.INTER_NEAREST)
📌 How it works:

Takes the closest pixel and copies its value.
No calculations are done—just picks the nearest pixel!

✅ Fastest method
❌ Produces pixelated images when enlarged.

🔹 Example:
Imagine zooming into a low-resolution image. The computer just duplicates nearby pixels, making it look blocky.

2️⃣ Bilinear Interpolation (cv2.INTER_LINEAR)
📌 How it works:

Looks at 4 nearest pixels and takes a weighted average of them.

This creates a smooth transition between pixels.

What is a Weighted Average? 🤔
A weighted average gives more importance to closer pixels and less importance to farther pixels.

🔹 Example:
Let’s say you want to estimate the brightness of a new pixel at (x, y).
If the nearby pixels have brightness 10, 20, 30, and 40, a simple average would be:

(10+20+30+40)/4=25

But in a weighted average, pixels that are closer get higher weight.
If the weights are 0.4, 0.3, 0.2, 0.1, the calculation becomes:

(10×0.4)+(20×0.3)+(30×0.2)+(40×0.1)=18

This creates smoother images by giving more importance to closer pixels.

✅ Faster than cubic interpolation
✅ Good for general-purpose upscaling and downscaling
❌ Can cause slight blurring when reducing image size.

3️⃣ Bicubic Interpolation (cv2.INTER_CUBIC)
📌 How it works:

Uses a cubic function to look at 16 nearest pixels and calculate new pixel values.
Produces sharper, smoother images than bilinear interpolation.

What is a Cubic Function? 🤔
A cubic function is a type of mathematical equation that follows the form:

𝑓(𝑥)=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑

This function creates a smoother curve compared to a straight line (linear) or a simple weighted average.
It ensures that the gradients between pixels change smoothly.

🔹 Example:
Think of it like drawing a curved line instead of straight lines between points. The cubic function helps to smooth the transitions without sharp edges.

✅ Better than bilinear interpolation
✅ Smooth, high-quality scaling
❌ Slower than bilinear

4️⃣ Lanczos Interpolation (cv2.INTER_LANCZOS4)
📌 How it works:

Uses an advanced mathematical function called the Lanczos kernel to estimate pixel values.
It takes 8×8 neighboring pixels (instead of just 4 or 16 like bilinear and bicubic).
Provides the highest quality for downscaling (reducing image size).
What is the Advanced Mathematical Function Used in Lanczos? 🤔
Lanczos interpolation uses the sinc function:

sinc(x) = sin(πx)/πx
​
The sinc function is used because it removes high-frequency noise and keeps sharp edges intact.
The Lanczos filter modifies this function by applying a window function, which makes it work better in image processing.
🔹 Example:
Imagine reducing the size of a high-resolution photo.

Lanczos ensures that sharp edges remain sharp instead of becoming blurry.
It does this by carefully removing unnecessary pixels while preserving the overall shape.

✅ Best quality for reducing image size
✅ Keeps sharp details intact
❌ Very slow because it involves complex calculations.

5️⃣ Area-Based Interpolation (cv2.INTER_AREA)
📌 How it works:

Instead of averaging pixels like bilinear or cubic methods, it calculates the average pixel value in an area and assigns it to the new pixel.
🔹 Example:
If an image is shrunk to half its size, each new pixel will be the average of 4 original pixels.

✅ Best for reducing image size
✅ Preserves details better than bilinear
❌ Not good for enlarging images

Which One Should You Use? 🤔
1️⃣ If you need speed:
👉 Use INTER_NEAREST (but expect pixelation).

2️⃣ If you want a balance between speed and quality:
👉 Use INTER_LINEAR (good general-purpose method). (USED IN THIS PROJECT)

3️⃣ If you want better quality than bilinear:
👉 Use INTER_CUBIC (smooth and high quality).

4️⃣ If you need the best quality, especially for downscaling:
👉 Use INTER_LANCZOS4 (sharp and detailed, but slow).

5️⃣ If you are downscaling (reducing image size) and want the best results:
👉 Use INTER_AREA (best for shrinking images).

