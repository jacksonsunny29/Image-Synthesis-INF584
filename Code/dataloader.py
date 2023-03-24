import torch 
import numpy as np 
import os



#Azimuthal Roation matrix
def rotation_theta_mat(t):
    rtm = torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]
    ]).float()

    return rtm


#Polar Roation matrix
def roation_phi_mat(p):
    rtm = torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]
    ]).float()

    return rtm

#Translation Matrix
def translation_mat(t):
   trans_mat =([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]
    ])

    return trans_mat


#Camera to World Tranformation.
#Will return a 4x4 transformation matrix
def cam_to_world(t,p,r):
    w = trans_t(r)
    w = rot_phi(p/180.*np.pi) @ w
    w = rot_theta(t/180.*np.pi) @ w
    w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ w
    return c2w

# Ray helpers
def get_rays(H, W, K, c2w):
    device = c2w.device

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                          torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# Ray helpers
def get_uvs_from_ray(H, W, K, c2w,pts):
    RP = torch.bmm(c2w[:3,:3].T[None,:,:].repeat(pts.shape[0],1,1),pts[:,:,None])[:,:,0]
    t = torch.mm(c2w[:3,:3].T,-c2w[:3,-1][:,None])
    pts_local0 = torch.sum((pts-c2w[:3,-1])[..., None, :] * (c2w[:3,:3].T), -1)
    pts_local = pts_local0/(-pts_local0[...,-1][...,None]+1e-7)
    u = pts_local[...,0]*K[0][0]+K[0][2]
    v = -pts_local[...,1]*K[1][1]+K[1][2]
    uv = torch.stack((u,v),-1)
    return uv,pts_local0


def batch_get_uv_from_ray(H,W,K,poses,pts):
    RT = (poses[:, :3, :3].transpose(1, 2))
    pts_local = torch.sum((pts[..., None, :] - poses[:, :3, -1])[..., None, :] * RT, -1)
    pts_local = pts_local / (-pts_local[..., -1][..., None] + 1e-7)
    u = pts_local[..., 0] * K[0][0] + K[0][2]
    v = -pts_local[..., 1] * K[1][1] + K[1][2]
    uv0 = torch.stack((u, v), -1)
    uv0[...,0] = uv0[...,0]/W*2-1
    uv0[...,1] = uv0[...,1]/H*2-1
    uv0 = uv0.permute(2,0,1,3)
    return uv0

def load(dir):
    datas = ['train', 'val', 'test']
    tr_data = {}
    for i in datas:
        with open(os.path.join(dir, 'transforms_{}.json'.format(s)), 'r') as x:
            tr_data[i] = json.load(x)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        poses = np.array(poses).astype(np.float32)
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    

    #H = Height , W = Width of the image sensor
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
    return imgs, poses, render_poses, [H, W, focal]


class Data_Def():
    def _init_(self,args):
        self.dataname = args.dataname
        self.datadir = os.path.join(args.datadir,args.dataname)
        self.logpath = self

        images, poses, render_poses, hwf = load(self.datadir)
        
        self.masks = images[..., -1:]
        self.images = images[..., :3]
        self.poses = poses
        self.render_poses = render_poses

        H, W, F = hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]


        #Camera Intrinsic Matrix Definition
        self.K = np.array([
                [F, 0, W/2.0],
                [0, F, H/2.0],
                [0, 0, 1]
        ])


    #Generation of Point Cloud
    def genpc(self):
        [H, W, focal] = self.hwf
        K = torch.tensor(self.K).cuda()
        train_n = 100
        poses = torch.tensor(self.poses).cuda()[:train_n]
        images = torch.tensor(self.images)[:train_n]

        pc,color,N = [],[],400
        [xs,ys,zs],[xe,ye,ze] = [-2,-2,-2],[2,2,2]
        pts_all = []
        for h_id in range(N):
            i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(),
                                  torch.linspace(ys, ye, N).cuda())  # pytorch's meshgrid has indexing='ij'
            i, j = i.t(), j.t()
            pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)
            pts[...,2] = h_id / N * (ze - zs) + zs
            pts_all.append(pts.clone())
            uv = batch_get_uv_from_ray(H,W,K,poses,pts)
            result = F.grid_sample(images.permute(0, 3, 1, 2).float(), uv).permute(0,2,3,1)

            margin = 0.05
            result[(uv[..., 0] >= 1.0) * (uv[..., 0] <= 1.0 + margin)] = 1
            result[(uv[..., 0] >= -1.0 - margin) * (uv[..., 0] <= -1.0)] = 1
            result[(uv[..., 1] >= 1.0) * (uv[..., 1] <= 1.0 + margin)] = 1
            result[(uv[..., 1] >= -1.0 - margin) * (uv[..., 1] <= -1.0)] = 1
            result[(uv[..., 0] <= -1.0 - margin) + (uv[..., 0] >= 1.0 + margin)] = 0
            result[(uv[..., 1] <= -1.0 - margin) + (uv[..., 1] >= 1.0 + margin)] = 0

            img = ((result>0.).sum(0)[...,0]>train_n-1).float()
            pc.append(img)
            color.append(result.mean(0))
        pc = torch.stack(pc,-1)
        color = torch.stack(color,-1)
        r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
        idx = torch.where(pc > 0)
        color = torch.stack((r[idx],g[idx],b[idx]),-1)
        idx = (idx[1],idx[0],idx[2])
        pts = torch.stack(idx,-1).float()/N
        pts[:,0] = pts[:,0]*(xe-xs)+xs
        pts[:,1] = pts[:,1]*(ye-ys)+ys
        pts[:,2] = pts[:,2]*(ze-zs)+zs

        pts = torch.cat((pts,color),-1).cpu().data.numpy()
        print('Initialization, data:{} point:{}'.format(self.dataname,pts.shape))
        item = MemDataset(pts,self.poses,self.images,self.masks,self.K)
        return item


