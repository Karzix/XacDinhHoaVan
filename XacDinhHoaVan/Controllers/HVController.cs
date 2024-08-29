using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;

public class HVController : ControllerBase
{
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
        int radius = (int)(0.4 * src.Width);

        // Tạo mặt nạ hình tròn
        Mat circleMask = new Mat(src.Size(), MatType.CV_8UC1, new Scalar(0));
        Cv2.Circle(circleMask, center, radius, new Scalar(255), -1);

        // Lọc chỉ các pixel đen trong vùng hình tròn
        Mat blackPixelsInCircle = new Mat();
        Cv2.BitwiseAnd(mask, circleMask, blackPixelsInCircle);

        // Đếm số lượng điểm ảnh đen
        int blackPixelCount = Cv2.CountNonZero(blackPixelsInCircle);

        // Trả về số lượng điểm ảnh đen
        return Ok(new { BlackPixelCount = blackPixelCount });
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
        int radius = (int)(0.4 * src.Width);

        // Vẽ hình tròn màu đỏ
        Cv2.Circle(src, center, radius, new Scalar(0, 0, 255), 2); // Màu đỏ (BGR: 0, 0, 255), độ dày 2

        // Mã hóa hình ảnh đã xử lý thành chuỗi base64
        byte[] processedImageBytes = src.ToBytes(".png");
        string base64String = System.Convert.ToBase64String(processedImageBytes);

        return Ok(base64String);
    }

}
