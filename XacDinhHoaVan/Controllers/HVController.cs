using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;

public class HVController : ControllerBase
{
    private double tiLeHinhTron = 0.35; 
    [HttpPost("check-pattern")]
    public IActionResult CheckImage(IFormFile imageFile)
    {
        
        if (imageFile == null || imageFile.Length == 0)
            return BadRequest("Invalid image file.");

        using var ms = new MemoryStream();
        imageFile.CopyTo(ms);
        var imageData = ms.ToArray();

        // Tải hình ảnh
        Mat src = Cv2.ImDecode(imageData, ImreadModes.Color);
        // Định nghĩa giới hạn dưới và trên cho dải màu từ đen đến xám đậm để giữ lại
        Scalar lowerBound = new Scalar(0, 0, 0); // Màu đen
        Scalar upperBound = new Scalar(149, 149, 149); // Màu xám đậm

        // Tạo mặt nạ cho các pixel nằm trong dải màu
        Mat mask = new Mat();
        Cv2.InRange(src, lowerBound, upperBound, mask);

        // Đặt các pixel ngoài dải màu thành màu đen
        src.SetTo(new Scalar(0, 0, 0), mask);

        // Xác định tâm và bán kính của hình tròn
        Point center = new Point(src.Width / 2, src.Height / 2);
        int radius = (int)(tiLeHinhTron * src.Width);

        // Tạo mặt nạ hình tròn
        Mat circleMask = new Mat(src.Size(), MatType.CV_8UC1, new Scalar(0));
        Cv2.Circle(circleMask, center, radius, new Scalar(255), -1);

        // Lọc chỉ các pixel đen trong vùng hình tròn
        Mat blackPixelsInCircle = new Mat();
        Cv2.BitwiseAnd(mask, circleMask, blackPixelsInCircle);

        // Đếm số lượng điểm ảnh đen
        int blackPixelCount = Cv2.CountNonZero(blackPixelsInCircle);

        // Trả về số lượng điểm ảnh đen
        return Ok(blackPixelCount);
    }
    [HttpPost("process-image")]
    public IActionResult ProcessImage(IFormFile imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
            return BadRequest("Invalid image file.");

        using var ms = new MemoryStream();
        imageFile.CopyTo(ms);
        var imageData = ms.ToArray();

        // Load image
        Mat src = Cv2.ImDecode(imageData, ImreadModes.Color);

        // Define the lower and upper bounds for the color range to keep
        Scalar lowerBound = new Scalar(149, 149, 149); // #dddddd
        Scalar upperBound = new Scalar(255, 255, 255); // #ffffff

        // Create a mask for pixels within the color range
        Mat mask = new Mat();
        Cv2.InRange(src, lowerBound, upperBound, mask);

        // Invert the mask to get the regions that are NOT in the color range
        Mat invertedMask = new Mat();
        Cv2.BitwiseNot(mask, invertedMask);

        // Set the pixels outside the color range to black
        src.SetTo(new Scalar(0, 0, 0), invertedMask);

        // Encode the processed image to base64
        byte[] processedImageBytes = src.ToBytes(".png");
        string base64String = System.Convert.ToBase64String(processedImageBytes);

        return Ok(base64String);
    }
    [HttpPost("draw-circle")]
    public IActionResult DrawCircle(IFormFile imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
            return BadRequest("Invalid image file.");

        using var ms = new MemoryStream();
        imageFile.CopyTo(ms);
        var imageData = ms.ToArray();

        // Tải hình ảnh
        Mat src = Cv2.ImDecode(imageData, ImreadModes.Color);

        // Xác định tâm và bán kính của hình tròn
        Point center = new Point(src.Width / 2, src.Height / 2);
        int radius = (int)(tiLeHinhTron * src.Width);

        // Vẽ hình tròn màu đỏ
        Cv2.Circle(src, center, radius, new Scalar(0, 0, 255), 2); // Màu đỏ (BGR: 0, 0, 255), độ dày 2

        // Mã hóa hình ảnh đã xử lý thành chuỗi base64
        byte[] processedImageBytes = src.ToBytes(".png");
        string base64String = System.Convert.ToBase64String(processedImageBytes);

        return Ok(base64String);
    }
    private bool IsLargeCircle(Mat src)
    {
        // Chuyển đổi ảnh sang ảnh xám
        Mat gray = new Mat();
        Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

        // Làm mờ ảnh để giảm nhiễu
        Mat blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(5, 5), 0);

        // Phát hiện các cạnh trong ảnh
        Mat edges = new Mat();
        Cv2.Canny(blurred, edges, 100, 200);

        // Tìm các đường viền trong ảnh
        Cv2.FindContours(edges, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        // Tính diện tích của toàn bộ ảnh
        double totalArea = src.Width * src.Height;

        // Duyệt qua các đường viền để kiểm tra hình tròn lớn
        foreach (var contour in contours)
        {
            // Tính chu vi của đường viền
            double perimeter = Cv2.ArcLength(contour, true);

            // Xấp xỉ hình dạng của đường viền
            Point[] approx = Cv2.ApproxPolyDP(contour, 0.04 * perimeter, true);

            // Tính diện tích của đường viền
            double area = Cv2.ContourArea(contour);

            // Tính tỷ lệ giữa diện tích và chu vi để xác định hình tròn
            double circularity = 4 * Math.PI * (area / (perimeter * perimeter));

            // Nếu tỷ lệ gần với 1, đó là hình tròn
            if (circularity > 0.8 && circularity <= 1.2)
            {
                // Tìm tâm và bán kính của hình tròn
                Cv2.MinEnclosingCircle(contour, out Point2f center, out float radius);

                // Tính diện tích của hình tròn
                double circleArea = Math.PI * radius * radius;

                // So sánh diện tích hình tròn với diện tích ảnh
                if ((int)circleArea > (int)(0.7 * totalArea))
                {
                    return true;
                }
            }
        }

        return false;
    }
    [HttpPost("RemoveBackground")]
    public IActionResult DrawRedOutlineAndCountBlackPixels(IFormFile imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
            return BadRequest("No image file provided.");

        try
        {
            // Đọc file hình ảnh từ input stream
            using var ms = new MemoryStream();
            imageFile.CopyTo(ms);
            byte[] fileBytes = ms.ToArray();

            Mat src = Cv2.ImDecode(fileBytes, ImreadModes.Color);

            // Chuyển đổi sang ảnh xám (ảnh đen trắng)
            Mat gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            // Chuyển đổi ảnh xám sang ảnh trắng đen (sử dụng threshold)
            Mat binary = new Mat();
            Cv2.Threshold(gray, binary, 128, 255, ThresholdTypes.Binary);

            // Sử dụng ảnh `binary` để xử lý thay cho `src`
            src = binary;

            // Tìm contour của đối tượng đĩa
            Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            // Tìm contour lớn nhất (giả sử đó là vật chủ)
            Point[] largestContour = null;
            double maxArea = 0;

            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                if (area > maxArea)
                {
                    maxArea = area;
                    largestContour = contour;
                }
            }

            if (largestContour == null)
                return BadRequest("No object detected.");

            // Tạo một mask để xác định vùng contour lớn nhất
            Mat mask = Mat.Zeros(src.Size(), MatType.CV_8UC1);
            Cv2.DrawContours(mask, new[] { largestContour }, -1, Scalar.White, -1);

            int blackPixelCountInObject = 0; // Đếm pixel đen trong vật chủ
            int blackPixelCountInsideCircle = 0; // Đếm pixel đen trong hình tròn 30%

            // Tạo một mask để xác định vùng contour lớn nhất
            Mat erodedMask = new Mat();
            Cv2.Erode(mask, erodedMask, Cv2.GetStructuringElement(MorphShapes.Rect, new Size(10, 10)));

            // Đếm số lượng điểm ảnh đen trong vật chủ
            Mat blackMask = new Mat();
            Cv2.InRange(src, new Scalar(0), new Scalar(50), blackMask); // Giới hạn màu để đếm pixel đen
            blackPixelCountInObject = Cv2.CountNonZero(blackMask & erodedMask);

            // Vẽ đường viền màu đỏ xung quanh contour lớn nhất
            Cv2.DrawContours(src, new[] { largestContour }, -1, new Scalar(0, 0, 255), 5);

            // Vẽ một hình tròn lớn bằng 30% kích thước vật chủ
            var rect = Cv2.BoundingRect(largestContour);
            int radius = (int)(Math.Min(rect.Width, rect.Height) * 0.45 / 2);
            Point center = new Point(rect.X + rect.Width / 2, rect.Y + rect.Height / 2);

            // Đếm số lượng điểm ảnh đen bên trong hình tròn
            Mat circleMask = Mat.Zeros(src.Size(), MatType.CV_8UC1);
            Cv2.Circle(circleMask, center, radius, Scalar.White, -1);

            // Đếm pixel đen bên trong hình tròn
            blackPixelCountInsideCircle = Cv2.CountNonZero(blackMask & circleMask);

            // Vẽ viền xung quanh hình tròn nơi đếm số pixel đen
            Cv2.Circle(src, center, radius, new Scalar(0, 0, 255), 2);

            // Chuyển ảnh kết quả sang base64
            byte[] resultBytes = src.ToBytes(".png");
            string base64String = Convert.ToBase64String(resultBytes);
            string loaidia = "";

            if (blackPixelCountInObject < 750)
            {
                loaidia = "dĩa trắng";
            }
            else if (blackPixelCountInObject - blackPixelCountInsideCircle < 750)
            {
                loaidia = "chỉ có tâm";
            }
            else if (blackPixelCountInObject > 750 && blackPixelCountInsideCircle < 750)
            {
                loaidia = "chỉ có viền";
            }
            else
            {
                loaidia = "có cả tâm với viền";
            }

            // Trả về ảnh và số lượng điểm ảnh đen trong vật chủ, và trong hình tròn
            return Ok(new
            {
                Image = base64String,
                BlackPixelCountInObject = blackPixelCountInObject,
                BlackPixelCountInsideCircle = blackPixelCountInsideCircle,
                LoaiDia = loaidia,
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error processing image: {ex.Message}");
        }
    }


}
