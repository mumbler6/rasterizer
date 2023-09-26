import sys
import math
from PIL import Image
import numpy as np
import copy

pixelLocs = []
pixelColors = []
elements = []
texcoords = []
depth_buffer = None
height = 0
width = 0
depth = False
sRGB = False
hyp = False
filename = ""
texture_image = None
image = None
matrix = None
decals = False

def DDA_I(point_a, point_b, dim_d):
    if point_a[dim_d] == point_b[dim_d]:
        return []
    if point_a[dim_d] > point_b[dim_d]:
        temp = point_a
        point_a = point_b
        point_b = temp

    delta = point_b - point_a
    point_s = delta / delta[dim_d]

    e_const = math.ceil(point_a[dim_d]) - point_a[dim_d]
    point_o = e_const * point_s
    point_p = point_a + point_o

    return [point_p, point_s]

def DDA_L(point_a, point_b, dim_d):
    if point_a[dim_d] == point_b[dim_d]:
        return []
    if point_a[dim_d] > point_b[dim_d]:
        temp = point_a
        point_a = point_b
        point_b = temp

    delta = point_b - point_a
    point_s = delta / delta[dim_d]

    e_const = math.ceil(point_a[dim_d]) - point_a[dim_d]
    point_o = e_const * point_s
    point_p = point_a + point_o

    return_list = []
    while (point_p[dim_d] < point_b[dim_d]):
        return_list.append(copy.deepcopy(point_p))
        point_p += point_s

    return return_list


def scanline(points):
    points1 = points[np.argsort(points[:,1])]
    point_t = points1[0]
    point_b = points1[2]
    point_m = points1[1]

    points_l = DDA_I(point_t, point_b, 1)
    if (len(points_l) == 0):
        return []
    
    point_pl = points_l[0]
    point_sl = points_l[1]

    points_in_triangle = []
    points_top = DDA_I(point_t, point_m, 1)
    if (len(points_top) != 0):
        point_p = points_top[0]
        point_s = points_top[1]

        while point_p[1] < point_m[1]:
            points_in = DDA_L(point_p, point_pl, 0)
            points_in_triangle.extend(points_in)
            point_p += point_s
            point_pl += point_sl

    points_bottom = DDA_I(point_m, point_b, 1)
    if (len(points_bottom) != 0):
        point_p = points_bottom[0]
        point_s = points_bottom[1]
        while point_p[1] < point_b[1]:
            points_in = DDA_L(point_p, point_pl, 0)
            points_in_triangle.extend(points_in)
            point_p += point_s
            point_pl += point_sl

    return points_in_triangle

def createBig(big_vector):
    for i in range(len(pixelLocs)):
        big_vector.append(copy.deepcopy(pixelLocs[i]))
        if len(pixelColors) > 0:
            big_vector[i].extend(copy.deepcopy(pixelColors[i]))
        else:
            big_vector[i].extend([0.0,0.0,0.0,1.0])
        if len(texcoords) > 0:
            big_vector[i].extend(copy.deepcopy(texcoords[i]))
        else:
            big_vector[i].extend([0.0,0.0])

def transform(pixels):
    for pixel in pixels:
        pixel[0] = (pixel[0]/pixel[3]+1)*width/2
        pixel[1] = (pixel[1]/pixel[3]+1)*height/2

        if hyp:
            pixel[2] /= pixel[3]
            for i in range(4, len(pixel)):
                pixel[i] /= pixel[3]
            pixel[3] = 1.0 / pixel[3]

def matrix_transform(pixels):
    for point in pixels:
        point[0:4] = matrix @ point[0:4]

def hyp_transform(pixels):
    for pixel in pixels:
        for i in range(4, len(pixel)):
            pixel[i] /= pixel[3]
        pixel[3] = 1.0 / pixel[3]

def to_sRGB(point):
    for i in range(4, 7):
        if point[i] <= 0.0031308:
            point[i] = 12.92 * point[i]
        elif point[i] > 0.0031308:
            point[i] = 1.055 * point[i] ** (1.0/2.4) - 0.055

def from_sRGB(point):
    for i in range(3):
        if point[i] <= 0.04045:
            point[i] /= 12.92
        elif point[i] > 0.04045:
            point[i] = ((point[i] + 0.055) / 1.055) ** 2.4

def draw(points_in_triangle):
    for point in points_in_triangle:
        if texture_image is not None:
            point[8] %= 1
            point[9] %= 1
            texture_image_width, texture_image_height = texture_image.shape[0], texture_image.shape[1]
            tex_y = int(point[8] * texture_image_width)
            tex_x = int(point[9] * texture_image_height)
            tex_color = texture_image[tex_x][tex_y] / 255
            if not decals:
                point[4:4 + texture_image.shape[2]] = tex_color
            else:
                from_sRGB(tex_color)
                point[4:7] = (tex_color[0:3]*tex_color[3] + point[4:7]*(1-tex_color[3]))
                point[7] = tex_color[3] + point[7] - tex_color[3] * point[7]
                to_sRGB(point)

        x_coord = int(point[0])
        y_coord = int(point[1])
        if x_coord >= 0 and x_coord < width and y_coord >= 0 and y_coord < height:
            if sRGB and texture_image is None:
                to_sRGB(point)
            
            if depth:
                if point[2] < depth_buffer[x_coord][y_coord]:
                    depth_buffer[x_coord][y_coord] = point[2]
                    image.putpixel([x_coord, y_coord], tuple((point[4:8]*255).astype(int)))
            else:
                image.putpixel([x_coord, y_coord], tuple((point[4:8]*255).astype(int)))

    image.save(filename)

def main():
    arg = sys.argv[1]
    file = open(arg, 'r')

    for line in file:
        if (line.isspace()):
            continue

        line = line.split()
        keyword = line[0]
        
        if keyword == "png":
            global width, height, image, filename
            width = int(line[1])
            height = int(line[2])
            image = Image.new("RGBA", (width, height), (0,0,0,0))
            filename = line[3]
            image.save(filename)
            
        elif keyword == "position":
            global pixelLocs
            pixelLocs = []
            size = int(line[1])
            transformed = False
            for i in range(2, len(line), size):
                vector = []
                for j in range(i, i + size):
                    vector.append(float(line[j]))
                if size < 3:
                    vector.append(0)
                    vector.append(1)
                elif size < 4:
                    vector.append(1)
                pixelLocs.append(vector)

        elif keyword == "texcoord":
            global texcoords
            texcoords = []
            size = int(line[1])
            for i in range(2, len(line), 2):
                vector = []
                vector.append(float(line[i]))
                vector.append(float(line[i + 1]))
                texcoords.append(vector)
    
        elif keyword == "texture":
            global texture_image
            text_filename = line[1]
            texture_image = np.asarray(Image.open(text_filename))

        elif keyword == "color":
            global pixelColors
            pixelColors = []
            size = int(line[1])
            if size == 3:
                for i in range(2, len(line), size):
                    vector = []
                    for j in range(i, i + size):
                        vector.append(float(line[j]))
                    vector.append(1)
                    pixelColors.append(vector)
            elif size == 4:
                for i in range(2, len(line), size):
                    vector = []
                    for j in range(i, i + size):
                        vector.append(float(line[j]))
                    pixelColors.append(vector)
        
        elif keyword == "elements":
            global elements
            elements = []
            for i in range(1, len(line)):
                elements.append(int(line[i]))
            
        elif keyword == "depth":
            global depth_buffer, depth
            depth_buffer = np.ones((width, height)) * np.inf
            depth = True
        
        elif keyword == "sRGB":
            global sRGB
            sRGB = True
        
        elif keyword == "decals":
            global decals
            decals = True

        elif keyword == "hyp":
            global hyp
            hyp = True

        elif keyword == "uniformMatrix":
            global matrix
            matrix = np.empty((4,4))
            index = 1
            for col in range(4):
                for row in range(4):
                    matrix[row][col] = float(line[index])
                    index += 1

        elif keyword == "drawArraysTriangles": 
            big_vector = []
            first = int(line[1])
            count = int(line[2])
            createBig(big_vector)
            if matrix is not None:
                matrix_transform(big_vector)
            transform(big_vector)
            
            for i in range(first, count, 3):
                points = big_vector[i:i+3]
                points_in_triangle = scanline(np.array(points))
                if hyp:
                    hyp_transform(points_in_triangle)
                draw(points_in_triangle)

        elif keyword == "drawElementsTriangles":
            big_vector = []
            count = int(line[1])
            offset = int(line[2])
            createBig(big_vector)
            if matrix is not None:
                matrix_transform(big_vector)
            transform(big_vector)
            for i in range(0, count, 3):
                ele = elements[offset+i:offset+i+3]
                points = [big_vector[ele[0]], big_vector[ele[1]], big_vector[ele[2]]]
                points_in_triangle = scanline(np.array(points))
                if hyp:
                    hyp_transform(points_in_triangle)
                draw(points_in_triangle)
            
if __name__ == "__main__":
    main()