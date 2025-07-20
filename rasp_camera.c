/*
 * QNX Camera UDP Streamer - Simple UDP client that streams camera frames
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <termios.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <camera/camera_api.h>

// UDP configuration - send to Python server
#define SERVER_IP "10.33.53.104"  // IP where Python runs
#define SERVER_PORT 9999

// Global UDP socket
static int g_udp_socket = -1;
static struct sockaddr_in g_server_addr;

// Camera frame header structure
typedef struct {
    uint32_t magic;       // 0x46524D45 ("FRME")
    uint32_t sequence;    // Frame number
    uint32_t frametype;   // QNX camera frametype
    uint32_t width;       // Frame width
    uint32_t height;      // Frame height
    uint32_t stride;      // Bytes per line
    uint32_t data_size;   // Frame data size
} frame_header_t;

// Supported frametypes
const camera_frametype_t cSupportedFrametypes[] = {
    CAMERA_FRAMETYPE_YCBYCR,
    CAMERA_FRAMETYPE_CBYCRY,
    CAMERA_FRAMETYPE_RGB8888,
    CAMERA_FRAMETYPE_BGR8888,
};
#define NUM_SUPPORTED_FRAMETYPES (sizeof(cSupportedFrametypes) / sizeof(cSupportedFrametypes[0]))

static void listAvailableCameras(void);
static int initUdpSocket(void);
static void closeUdpSocket(void);
static void processCameraData(camera_handle_t handle, camera_buffer_t* buffer, void* arg);
static void blockOnKeyPress(void);

int main(int argc, char* argv[])
{
    int err;
    int opt;
    camera_unit_t unit = CAMERA_UNIT_NONE;
    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    camera_frametype_t frametype = CAMERA_FRAMETYPE_UNSPECIFIED;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "u:")) != -1) {
        switch (opt) {
        case 'u':
            unit = (camera_unit_t)strtol(optarg, NULL, 10);
            break;
        default:
            printf("Usage: %s -u <camera_unit>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // Check if camera unit specified
    if (unit == CAMERA_UNIT_NONE || unit >= CAMERA_UNIT_NUM_UNITS) {
        listAvailableCameras();
        printf("Please specify camera unit with -u option\n");
        exit(EXIT_SUCCESS);
    }

    // Initialize UDP socket
    printf("Setting up UDP connection to %s:%d...\n", SERVER_IP, SERVER_PORT);
    if (initUdpSocket() != 0) {
        printf("Failed to initialize UDP socket.\n");
        exit(EXIT_FAILURE);
    }

    // Open camera
    err = camera_open(unit, CAMERA_MODE_RO, &handle);
    if (err != CAMERA_EOK || handle == CAMERA_HANDLE_INVALID) {
        printf("Failed to open camera unit %d: err = %d\n", unit, err);
        closeUdpSocket();
        exit(EXIT_FAILURE);
    }

    // Check frametype
    err = camera_get_vf_property(handle, CAMERA_IMGPROP_FORMAT, &frametype);
    if (err != CAMERA_EOK) {
        printf("Failed to get frametype: err = %d\n", err);
        camera_close(handle);
        closeUdpSocket();
        exit(EXIT_FAILURE);
    }

    // Verify supported frametype
    bool supported = false;
    for (uint i = 0; i < NUM_SUPPORTED_FRAMETYPES; i++) {
        if (frametype == cSupportedFrametypes[i]) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        printf("Unsupported frametype: %d\n", frametype);
        camera_close(handle);
        closeUdpSocket();
        exit(EXIT_FAILURE);
    }

    printf("Camera opened. Starting stream...\n");
    printf("Press any key to stop.\n\n");

    // Start streaming
    err = camera_start_viewfinder(handle, processCameraData, NULL, NULL);
    if (err != CAMERA_EOK) {
        printf("Failed to start camera: err = %d\n", err);
        camera_close(handle);
        closeUdpSocket();
        exit(EXIT_FAILURE);
    }

    // Wait for user input
    blockOnKeyPress();

    // Cleanup
    camera_stop_viewfinder(handle);
    camera_close(handle);
    closeUdpSocket();
    
    printf("\nStopped.\n");
    return 0;
}

static void listAvailableCameras(void)
{
    int err;
    uint numSupported;
    camera_unit_t* supportedCameras;

    err = camera_get_supported_cameras(0, &numSupported, NULL);
    if (err != CAMERA_EOK) {
        printf("Failed to get camera count: err = %d\n", err);
        return;
    }

    if (numSupported == 0) {
        printf("No cameras detected!\n");
        return;
    }

    supportedCameras = calloc(numSupported, sizeof(camera_unit_t));
    if (!supportedCameras) {
        printf("Memory allocation failed\n");
        return;
    }

    err = camera_get_supported_cameras(numSupported, &numSupported, supportedCameras);
    if (err == CAMERA_EOK) {
        printf("Available cameras:\n");
        for (uint i = 0; i < numSupported; i++) {
            printf("  Camera %d (use -u %d)\n", supportedCameras[i], supportedCameras[i]);
        }
    }

    free(supportedCameras);
}

static int initUdpSocket(void)
{
    // Create UDP socket
    g_udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_udp_socket < 0) {
        printf("UDP socket creation failed: %s\n", strerror(errno));
        return -1;
    }

    // Setup server address
    memset(&g_server_addr, 0, sizeof(g_server_addr));
    g_server_addr.sin_family = AF_INET;
    g_server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, SERVER_IP, &g_server_addr.sin_addr) <= 0) {
        printf("Invalid server IP: %s\n", SERVER_IP);
        close(g_udp_socket);
        g_udp_socket = -1;
        return -1;
    }

    printf("UDP socket ready!\n");
    return 0;
}

static void closeUdpSocket(void)
{
    if (g_udp_socket >= 0) {
        close(g_udp_socket);
        g_udp_socket = -1;
    }
}

static void processCameraData(camera_handle_t handle, camera_buffer_t* buffer, void* arg)
{
    static uint32_t frame_sequence = 0;
    static uint32_t frame_skip_counter = 0;
    frame_header_t header;
    uint32_t width, height, stride;
    uint32_t crop_width, crop_height;
    uint32_t reduced_width, reduced_height, reduced_stride, reduced_data_size;
    uint32_t bytes_per_pixel;
    uint32_t crop_x_start, crop_y_start;

    (void)handle;
    (void)arg;

    if (g_udp_socket < 0) return;

    // Skip frames for worse temporal quality (send every 16th frame for maximum compression)
    // frame_skip_counter++;
    // if (frame_skip_counter % 2 != 0) {
    //     return;
    // }

    // Extract frame info based on type
    switch (buffer->frametype) {
    case CAMERA_FRAMETYPE_RGB8888:
        width = buffer->framedesc.rgb8888.width;
        height = buffer->framedesc.rgb8888.height;
        stride = buffer->framedesc.rgb8888.stride;
        bytes_per_pixel = 4;
        break;
    case CAMERA_FRAMETYPE_BGR8888:
        width = buffer->framedesc.bgr8888.width;
        height = buffer->framedesc.bgr8888.height;
        stride = buffer->framedesc.bgr8888.stride;
        bytes_per_pixel = 4;
        break;
    case CAMERA_FRAMETYPE_YCBYCR:
        width = buffer->framedesc.ycbycr.width;
        height = buffer->framedesc.ycbycr.height;
        stride = buffer->framedesc.ycbycr.stride;
        bytes_per_pixel = 2;
        break;
    case CAMERA_FRAMETYPE_CBYCRY:
        width = buffer->framedesc.cbycry.width;
        height = buffer->framedesc.cbycry.height;
        stride = buffer->framedesc.cbycry.stride;
        bytes_per_pixel = 2;
        break;
    default:
        return;
    }

    // Extreme compression with worse quality - crop to tiny center and heavy downsample
    uint32_t min_dimension = (width < height) ? width : height;
    
    // Crop to center 1/16 of the image for MAXIMUM compression (even smaller crop)
    crop_width = min_dimension / 1;
    crop_height = min_dimension / 1;
    
    // Center the crop
    crop_x_start = (width - crop_width) / 2;
    crop_y_start = (height - crop_height) / 2;
    
    // EXTREME downsampling with terrible quality
    uint32_t downsample_factor = 16;  // Much more aggressive downsampling
    reduced_width = crop_width / downsample_factor;
    reduced_height = crop_height / downsample_factor;
    
    // Set minimum to 128x128 for maximum compression
    if (reduced_width < 512) reduced_width = 512;
    if (reduced_height < 512) reduced_height = 512;

    // Convert to GRAYSCALE to reduce data by 75% (1 byte per pixel instead of 4)
    // BUT use 4-bit quantization for MAXIMUM compression
    uint32_t compressed_bytes_per_pixel = 1;  // Still 1 byte but with 4-bit precision
    reduced_stride = reduced_width * compressed_bytes_per_pixel;
    reduced_data_size = reduced_stride * reduced_height;

    // Prepare header with reduced dimensions
    header.magic = 0x46524D45;
    header.sequence = frame_sequence++;
    header.frametype = 99;  // Custom frametype for grayscale (1 byte per pixel)
    header.width = reduced_width;
    header.height = reduced_height;
    header.stride = reduced_stride;
    header.data_size = reduced_data_size;

    // Send header via UDP
    if (sendto(g_udp_socket, &header, sizeof(header), 0, 
               (struct sockaddr*)&g_server_addr, sizeof(g_server_addr)) != sizeof(header)) {
        printf("\rHeader send failed");
        return;
    }

    // Send compressed frame data via UDP in chunks
    uint8_t* source_data = (uint8_t*)buffer->framebuf;
    ssize_t total_sent = 0;
    const uint32_t MAX_UDP_CHUNK = 3000;  // Safe UDP payload size
    
    // Calculate sampling rate for maximum compression
    uint32_t row_step = crop_height / reduced_height;
    uint32_t col_step = crop_width / reduced_width;
    
    // Ensure minimum step of 1 (no upsampling)
    if (row_step < 1) row_step = 1;
    if (col_step < 1) col_step = 1;
    
    // Prepare frame data buffer
    uint8_t* frame_buffer = malloc(reduced_data_size);
    if (!frame_buffer) {
        printf("\rMemory allocation failed");
        return;
    }
    
    uint32_t buffer_pos = 0;
    
    // Collect compressed frame data with quality degradation
    for (uint32_t row = 0; row < reduced_height; row++) {
        uint32_t source_row = crop_y_start + (row * row_step);
        // Bounds check
        if (source_row >= height) source_row = height - 1;
        
        uint8_t* row_start = source_data + (source_row * stride) + (crop_x_start * bytes_per_pixel);
        
        // Collect compressed pixels with aggressive compression to GRAYSCALE
        for (uint32_t col = 0; col < reduced_width; col++) {
            uint32_t source_col = col * col_step * bytes_per_pixel;
            // Bounds check
            if (crop_x_start * bytes_per_pixel + source_col >= stride) {
                source_col = stride - crop_x_start * bytes_per_pixel - bytes_per_pixel;
            }
            
            // Convert to GRAYSCALE with 8x8 block averaging for MAXIMUM compression
            uint32_t luminance_sum = 0;
            uint32_t pixel_count = 0;
            
            // Sample 8x8 block and convert to grayscale for MAXIMUM quality loss
            for (int dy = 0; dy < 8; dy++) {
                for (int dx = 0; dx < 8; dx++) {
                    uint32_t sample_row = source_row + dy;
                    uint32_t sample_col_offset = source_col + (dx * bytes_per_pixel);
                    
                    // Improved bounds checking
                    if (sample_row < height && 
                        crop_x_start * bytes_per_pixel + sample_col_offset + bytes_per_pixel <= stride) {
                        
                        uint8_t* sample_start = source_data + (sample_row * stride) + 
                                              (crop_x_start * bytes_per_pixel) + sample_col_offset;
                        
                        uint32_t luminance;
                        if (bytes_per_pixel == 4) {
                            // RGB/BGR to grayscale using ITU-R BT.709 standard (lossy conversion)
                            uint32_t r = sample_start[0];
                            uint32_t g = sample_start[1]; 
                            uint32_t b = sample_start[2];
                            luminance = (77 * r + 150 * g + 29 * b) >> 8;  // Fast integer math
                        } else {
                            // YUV: Just take Y (luminance) component
                            luminance = sample_start[0];
                        }
                        
                        luminance_sum += luminance;
                        pixel_count++;
                    }
                }
            }
            
            // Store single grayscale byte (75% size reduction from RGB)
            if (pixel_count > 0) {
                uint8_t grayscale_value = luminance_sum / pixel_count;
                // Apply EXTREME quantization to reduce quality further (4-bit precision!)
                grayscale_value = (grayscale_value >> 4) << 4;  // Reduce to 4-bit precision (16 levels only!)
                
                // Safety check to prevent buffer overflow
                if (buffer_pos < reduced_data_size) {
                    frame_buffer[buffer_pos++] = grayscale_value;
                }
            } else {
                // Fallback: convert first pixel to grayscale
                uint8_t* pixel = row_start + source_col;
                uint32_t luminance;
                if (bytes_per_pixel == 4) {
                    uint32_t r = pixel[0], g = pixel[1], b = pixel[2];
                    luminance = (77 * r + 150 * g + 29 * b) >> 8;
                } else {
                    luminance = pixel[0];
                }
                luminance = (luminance >> 4) << 4;  // Extreme 4-bit quantization
                
                // Safety check to prevent buffer overflow
                if (buffer_pos < reduced_data_size) {
                    frame_buffer[buffer_pos++] = (uint8_t)luminance;
                }
            }
        }
    }
    
    // Apply Run-Length Encoding (RLE) for additional compression
    uint8_t* rle_buffer = malloc(reduced_data_size * 2);  // Worst case: double size
    if (!rle_buffer) {
        printf("\rRLE buffer allocation failed");
        free(frame_buffer);
        return;
    }
    
    uint32_t rle_size = 0;
    uint32_t i = 0;
    
    while (i < buffer_pos) {
        uint8_t current_value = frame_buffer[i];
        uint32_t count = 1;
        
        // Count consecutive identical values
        while (i + count < buffer_pos && frame_buffer[i + count] == current_value && count < 255) {
            count++;
        }
        
        if (count > 3 || current_value == 0) {  // RLE beneficial for runs > 3 or zeros
            rle_buffer[rle_size++] = 0;  // RLE marker
            rle_buffer[rle_size++] = count;
            rle_buffer[rle_size++] = current_value;
        } else {
            // Raw data for short runs
            for (uint32_t j = 0; j < count; j++) {
                rle_buffer[rle_size++] = frame_buffer[i + j];
            }
        }
        
        i += count;
    }
    
    // Use RLE data if it's smaller
    uint8_t* final_buffer;
    uint32_t final_size;
    
    if (rle_size < buffer_pos) {
        final_buffer = rle_buffer;
        final_size = rle_size;
        // Update header for RLE compressed data
        header.data_size = final_size;
        header.frametype = 100;  // RLE compressed grayscale
        
        // Re-send header with updated size
        if (sendto(g_udp_socket, &header, sizeof(header), 0, 
                   (struct sockaddr*)&g_server_addr, sizeof(g_server_addr)) != sizeof(header)) {
            printf("\rHeader resend failed");
            free(frame_buffer);
            free(rle_buffer);
            return;
        }
    } else {
        final_buffer = frame_buffer;
        final_size = buffer_pos;
    }
    
    // Send frame data in UDP chunks
    uint32_t chunks_sent = 0;
    for (uint32_t offset = 0; offset < final_size; offset += MAX_UDP_CHUNK) {
        uint32_t chunk_size = (offset + MAX_UDP_CHUNK > final_size) ? 
                             (final_size - offset) : MAX_UDP_CHUNK;
        
        ssize_t sent = sendto(g_udp_socket, final_buffer + offset, chunk_size, 0,
                             (struct sockaddr*)&g_server_addr, sizeof(g_server_addr));
        if (sent <= 0) {
            printf("\rData chunk send failed");
            free(frame_buffer);
            free(rle_buffer);
            return;
        }
        total_sent += sent;
        chunks_sent++;
        
        // Small delay between chunks to avoid overwhelming the network
        usleep(100);  // 0.1ms delay
    }
    
    free(frame_buffer);
    free(rle_buffer);

    // Show progress with extreme compression and poor quality
    float total_reduction = (float)(width * height * bytes_per_pixel) / (final_size);
    float rle_compression = (float)buffer_pos / final_size;
    float target_fps = 2.0f;  // Very low FPS due to every 16th frame
    printf("\rFrame %u: %dx%d → crop %dx%d → %dx%d → RLE (%.0fx total, %.1fx RLE, ~%.0f FPS, %u bytes, EXTREME 4-BIT+RLE)    ", 
           header.sequence, width, height, crop_width, crop_height, 
           reduced_width, reduced_height, total_reduction, rle_compression, target_fps, final_size);
    fflush(stdout);
}

static void blockOnKeyPress(void)
{
    struct termios oldterm, newterm;
    char key;

    tcgetattr(STDIN_FILENO, &oldterm);
    newterm = oldterm;
    newterm.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSANOW, &newterm);
    
    read(STDIN_FILENO, &key, 1);
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldterm);
}
