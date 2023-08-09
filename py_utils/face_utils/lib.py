
import cv2, torch
import numpy as np
from py_utils.face_utils.umeyama import umeyama


mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)


mean_face_98_828 = np.array([[156, 219], [194, 192], [241, 181], [286, 186], [329, 196], [328, 218], [288, 220], [241, 211], [194, 220], [498, 196], [541, 186], [587, 181], [635, 192], [671, 220], [631, 218], [587, 211], [541, 216], [497,222], [414, 282], [414, 333], [414, 384], [414, 437], [360, 477], [383,  483], [414, 487], [444, 483], [469, 477], [220, 284], [241, 271], [270, 263], [295, 269], [336, 288], [301, 300], [276, 303], [242, 302], [495,  290], [533, 269], [558, 263], [586, 271], [607, 284], [586, 300], [553, 303], [528, 298], [299, 589], [339, 560], [388, 542], [414, 547], [438, 542], [488, 561], [528, 589], [485, 607], [452, 618], [414, 621], [377, 618], [344, 606], [311, 584], [379, 572], [414, 572], [449, 572], [519, 586], [447, 586], [414, 585], [380, 585], [270, 283], [558, 283],     [84, 318], [93, 360], [92, 405], [101, 450], [107, 487], [113, 524], [123, 560], [134, 598], [148, 638], [168, 672], [194, 701], [218, 720], [247, 739], [279, 754], [313, 772], [351, 785], [413, 790], [476, 785], [513, 773], [550, 755], [581, 740], [611, 720], [635, 701], [659, 672], [679, 640], [695, 598], [704, 562], [715, 524], [722, 488], [727, 450], [736, 407], [737, 360], [744, 318]], dtype=float)

mean_face_x = mean_face_98_828[:, 0]
mean_face_y = mean_face_98_828[:, 1]
mean_face_x /= 824.0
mean_face_y /= 824.0
meanface_98_1 = np.zeros(mean_face_98_828.shape)
meanface_98_1[:, 0] = mean_face_x
meanface_98_1[:, 1] = mean_face_y



SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6
SELECT_DETECT_LMK_IDX = None
SELECT_STD_LMK_IDX = None
SELECTED_LANDMARKS_2D = None

def bgr2rgb(image):
    if len(image.shape) == 4:
        return image[:, :, :,  [2, 1, 0]]
    elif len(image.shape) == 3:
        return image[:, :,  [2, 1, 0]]

def cv2tensor(img_cv):
    img_tensor = torch.from_numpy(bgr2rgb(img_cv)).float()
    img_tensor = img_tensor.div(255.0)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze_(0)
    return img_tensor


## setup subset of landmarks used to align faces
def process_input_markID(IDs):
    final_id = []
    dashes_include = IDs.split(',')
    for item in dashes_include:
        item = item.strip().split('-')
        if (len(item) == 1):
            final_id.append(int(str(item[0])))
        else:
            for i in range(int(str(item[0])), int(str(item[1])) + 1):
                final_id.append(i)
    return final_id


def choose_landmarks_to_align(select_str):
    selected_ids = process_input_markID(select_str)
    selected_ids -= 17
    selected_pts = landmarks_2D[selected_ids]
    return selected_pts

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(shape, landmarks):
    OVERLAY_POINTS = [
        LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54]
    ]
    im = np.zeros(shape, dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im


def bur_size(landmarks):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    return blur_amount


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def cut_head(imgs, point):
    h, w = imgs[0].shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 8
    delta_y = (y2 - y1) / 5
    delta_x = np.random.randint(delta_x)
    delta_y = np.random.randint(delta_y)
    x1_ = np.int(np.maximum(0, x1 - delta_x))
    x2_ = np.int(np.minimum(w-1, x2 + delta_x))
    y1_ = np.int(np.maximum(0, y1 - delta_y))
    y2_ = np.int(np.minimum(h-1, y2 + delta_y * 0.5))
    imgs_new = []
    for i, im in enumerate(imgs):
        im = im[y1_:y2_, x1_:x2_, :]
        imgs_new.append(im)
    locs = [x1_, y1_, x2_, y2_]
    return imgs_new, locs


def get_2d_aligned_face(image, mat, size, padding=[0, 0]):
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    return cv2.warpAffine(image, mat, (size + 2 * padding[0], size + 2 * padding[1]))


def get_face_loc(im, face_detector, scale=0):
    """ get face locations, color order of images is rgb """
    faces = face_detector(np.uint8(im), scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for i, d in enumerate(faces):
            try:
                face_list.append([d.left(), d.top(), d.right(), d.bottom()])
            except:
                face_list.append([d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()])
    return face_list


def get_all_face_mask(shape, face_list):
    """
    Remove face area in head by black mask or mean color
    :param im:
    :param face_detector:
    :param lmark_predictor:
    :param tag: 0: white mask, 1: guassian blur, 2: random noise
    :return:
    """
    mask = np.zeros(shape)
    for _, points in face_list:
        mask += np.int32(get_face_mask(shape[:2], points))

    mask = np.uint8(mask > 0)
    return mask


def select_detected_lmks(landmarks):
    return landmarks[SELECT_DETECT_LMK_IDX]


def align(im, face_detector, lmark_predictor, scale=0):
    # This version we handle all faces in view
    # channel order rgb
    im = np.uint8(im)
    faces = face_detector(im, scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for pred in faces:
            points = shape_to_np(lmark_predictor(im, pred))
            trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
            face_list.append([trans_matrix, points])
    # import pdb; pdb.set_trace()

    if len(faces) == 2:
        # face_list = [0, 0]
        if faces[0].left() > faces[1].left():
            te = face_list[0].copy()
            face_list[0] = face_list[1].copy()
            face_list[1] = te

    return face_list


def align_landmarks(landmarks, stop_index=98, scale=0):
    trans_matrix_list = []
    for one_lm in landmarks:
        assert one_lm.shape == meanface_98_1.shape
        points = one_lm
        trans_matrix = umeyama(points[0: stop_index], meanface_98_1[0: stop_index], True)[0:2]
        trans_matrix_list.append([trans_matrix, points[0: stop_index]])
    return trans_matrix_list


def get_img_lmark(im, face_detector, lmark_predictor, scale=0):
    # This version we handle all faces in view
    # channel order rgb
    im = np.uint8(im)
    faces = face_detector(im, scale)

    if faces is not None and len(faces) > 0:
        # print(len(faces))
        # for pred in faces:
        pred = faces[0]
        points = shape_to_np(lmark_predictor(im, pred))
        # trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
    else:
        return np.array([])
    if len(points) != 68:
        print('{} points!!!'.format(len(points)))
        return np.array([])

    return points


def get_aligned_face_and_landmarks(im, face_cache, aligned_face_size = 256, padding=(0, 0)):
    """
    get all aligned faces and landmarks of all images
    :param imgs: origin images
    :param fa: face_alignment package
    :return:
    """
    aligned_cur_shapes = []
    aligned_cur_im = []
    for mat, points in face_cache:
        # Get transform matrix
        aligned_face = get_2d_aligned_face(im, mat, aligned_face_size, padding)
        # Mapping landmarks to aligned face
        pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        pred_ = np.transpose(pred_)
        mat = mat * aligned_face_size
        mat[0, 2] += padding[0]
        mat[1, 2] += padding[1]
        aligned_pred = np.dot(mat, pred_)
        aligned_pred = np.transpose(aligned_pred[:2, :])
        aligned_cur_shapes.append(aligned_pred)
        aligned_cur_im.append(aligned_face)

    return aligned_cur_im, aligned_cur_shapes


def get_aligned_landmarks(face_cache, aligned_face_size = 256, padding=(0, 0)):
    aligned_cur_shapes = []
    for mat, points in face_cache:
        # Get transform matrix
        # Mapping landmarks to aligned face
        pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        pred_ = np.transpose(pred_)
        mat = mat * aligned_face_size
        mat[0, 2] += padding[0]
        mat[1, 2] += padding[1]
        aligned_pred = np.dot(mat, pred_)
        aligned_pred = np.transpose(aligned_pred[:2, :])
        aligned_cur_shapes.append(aligned_pred)

    return aligned_cur_shapes


def get_face_mask_v2(shape, landmarks_aligned, mat, size):
    OVERLAY_POINTS = [
        LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54]
    ]

    pts = []
    # draw contours around cheek
    # right side
    w = landmarks_aligned[48][0] - landmarks_aligned[17][0]
    h = landmarks_aligned[48][1] - landmarks_aligned[17][1]

    x = landmarks_aligned[17][0]
    for i in range(1, 5):
        x = x + i * (w / 15)
        y = landmarks_aligned[17][1] + i * (h / 5)
        pts.append([x, y])

    w = landmarks_aligned[26][0] - landmarks_aligned[54][0]
    h = landmarks_aligned[54][1] - landmarks_aligned[26][1]

    x = landmarks_aligned[26][0]
    for i in range(1, 5):
        x = x - i * (w / 15)
        y = landmarks_aligned[26][1] + i * (h / 5)
        pts.append([x, y])

    for group in OVERLAY_POINTS:
        pts = np.concatenate([pts, landmarks_aligned[group]], 0)

    tmp = np.concatenate([np.transpose(pts), np.ones([1, pts.shape[0]])], 0)
    # Transform back to original location
    ary = np.expand_dims(np.array([0, 0, 1]), axis=0)
    mat = np.concatenate([mat * size, ary], 0)
    pts_org = np.dot(np.linalg.inv(mat), tmp)
    pts_org = pts_org[:2, :]
    pts_org = np.transpose(pts_org)
    im = np.zeros(shape, dtype=np.float64)

    draw_convex_hull(im,
                     np.int32(pts_org),
                     color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im

def get_face_mask_v3(shape, landmarks_aligned, mat, size, padding):
    OVERLAY_POINTS = [
        LEFT_BROW_POINTS + RIGHT_BROW_POINTS
    ]

    outer_inner_ratio = 0.5

    pts = []
    # draw contours around cheek
    # right side

    LOW_LIPS = [48, 59, 58, 57, 56, 55, 54]
    LOW_CHIN = [5,  6,   7,  8,  9, 10, 11]
    for idx in range(len(LOW_CHIN)):
        new_pt = get_interpolation(outer=landmarks_aligned[LOW_CHIN[idx]], inner=landmarks_aligned[LOW_LIPS[idx]], outer_inner_ratio=outer_inner_ratio)
        pts.append(new_pt)

    ## lmk 1 and lmk 37
    new_pt = get_interpolation(landmarks_aligned[0], landmarks_aligned[36], outer_inner_ratio)
    pts.append(new_pt)
    pt1 = new_pt

    # lmk4, 49
    new_pt = get_interpolation(landmarks_aligned[3], landmarks_aligned[48], outer_inner_ratio)
    pts.append(new_pt)


    #lmk5, 49
    new_pt = get_interpolation(landmarks_aligned[4], landmarks_aligned[48], outer_inner_ratio)
    pts.append(new_pt)
    pt2 = new_pt

    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1]

    x = pt1[0]
    for i in range(1, 5):
        x = x + i * (w / 15)
        y = pt1[1] + i * (h / 5)
        pts.append([x, y])

    ## lmk 17 and lmk 46
    new_pt = get_interpolation(landmarks_aligned[16], landmarks_aligned[45], outer_inner_ratio)
    pts.append(new_pt)
    pt1 = new_pt

    #lmk 13, 55
    new_pt = get_interpolation(landmarks_aligned[12], landmarks_aligned[54], outer_inner_ratio)
    pts.append(new_pt)
    pt2 = new_pt

    #lmk 14, 55
    new_pt = get_interpolation(landmarks_aligned[13], landmarks_aligned[54], outer_inner_ratio)
    pts.append(new_pt)

    w = pt1[0] - pt2[0]
    h = pt2[1] - pt1[1]

    x = pt1[0]
    for i in range(1, 5):
        x = x - i * (w / 15)
        y = pt1[1] + i * (h / 5)
        pts.append([x, y])

    for group in OVERLAY_POINTS:
        pts = np.concatenate([pts, landmarks_aligned[group]], 0)

    tmp = np.concatenate([np.transpose(pts), np.ones([1, pts.shape[0]])], 0)
    # Transform back to original location
    ary = np.expand_dims(np.array([0, 0, 1]), axis=0)
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    mat = np.concatenate([mat, ary], 0)
    pts_org = np.dot(np.linalg.inv(mat), tmp)
    pts_org = pts_org[:2, :]
    pts_org = np.transpose(pts_org)
    im = np.zeros(shape, dtype=np.float64)

    draw_convex_hull(im,
                     np.int32(pts_org),
                     color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im


def get_interpolation(outer, inner, outer_inner_ratio):
    x0 = outer[0]
    y0 = outer[1]
    x1 = inner[0]
    y1 = inner[1]
    x = x0 + outer_inner_ratio * (x1 - x0)
    y = y0 + outer_inner_ratio * (y1 - y0)

    return [x, y]


def crop_eye(img, points):
    eyes_list = []

    left_eye = points[36:42, :]
    right_eye = points[42:48, :]

    eyes = [left_eye, right_eye]
    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])

        w = rp - lp
        h = bp - tp

        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(img.shape[1], rp + 0.25 * w))
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(img.shape[0], bp + 1.75 * h))

        eyes_list.append(img[tp_:bp_, lp_:rp_, :])
    return eyes_list
