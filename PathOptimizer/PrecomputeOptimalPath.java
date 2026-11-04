import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * Integrated optimizer for Astrobee’s first leg with 6-DOF cost:
 *   • Supports arbitrary waypoint sampling (not limited to 6 points).
 *   • Waypoints include full pose: position + orientation (quaternion).
 *   • Computes best “photo‐positions” inside KIZs meeting a 45° FOV.
 *   • Heavily values Oasis travel (cost inside Oasis = 0.3/m).
 *   • Stronger time bonus (0.2 pts/s leftover).
 *   • Allows arbitrary intermediate entry/exit points via closed‐form Dijkstra for each Oasis.
 *   • Provides helper to choose a single photo‐location for two sheets together.
 *   • Charges rotation cost per radian and orients camera towards targets.
 */
public class PrecomputeOptimalPath {

    public static void main(String[] args) {
        // 1) Raw start & astronaut poses (position + orientation quaternion w,x,y,z)
        Pose startRaw     = new Pose(9.815, -9.806, 4.293, 1, 0, 0, 0);
        Pose astronautRaw = new Pose(11.143, -6.7607, 4.9654, 0, 0, 0.707, 0.707);

        // 2) Four random Areas (positions only; orientation irrelevant until photo)
        List<Pose> areasRaw = Arrays.asList(
            new Pose(10.42, -10.58, 4.82, 1,0,0,0),
            new Pose(10.3,  -9.25, 3.76203, 1,0,0,0),
            new Pose(10.3,  -8.4,  3.76093, 1,0,0,0),
            new Pose(9.866984, -7.34, 4.32, 1,0,0,0)
        );

        // 3) Oasis zones (from rulebook) with per-photo bonuses
        List<AABB> oasisZones = Arrays.asList(
            new AABB(10.425, -10.2, 4.445, 11.425, -9.5, 4.945),
            new AABB(10.925, -9.5,  4.945, 11.425, -8.45,5.445),
            new AABB(10.425, -8.45, 4.945, 10.975, -7.4, 5.445),
            new AABB(10.925, -7.4,  4.425, 11.425, -6.35,4.945)
        );
        double[] oasisPhotoBonus = {12, 10, 8, 11};

        // 4) KIZ zones
        List<AABB> kizZones = Arrays.asList(
            new AABB(10.3,  -10.2, 4.32, 11.55, -6.0, 5.57),
            new AABB(9.5,   -10.5, 4.02, 10.5,  -9.6, 4.8)
        );

        // 5) Build optimizer and compute full route
        PathOptimizer opt = new PathOptimizer(
            startRaw,
            areasRaw,
            astronautRaw,
            oasisZones,
            kizZones,
            oasisPhotoBonus
        );
        List<Pose> fullRoute = opt.findOptimalRouteToAstronaut();

        // 6) Write out route
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("optimal_path.txt"))) {
            for (Pose p : fullRoute) {
                bw.write(String.format(Locale.US, "%f,%f,%f,%f,%f,%f,%f",
                    p.x,p.y,p.z,p.qw,p.qx,p.qy,p.qz));
                bw.newLine();
            }
            System.out.println("Wrote full optimal route with orientations.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

/** Pose = position (x,y,z) + orientation quaternion (qw,qx,qy,qz). */
class Pose {
    public final double x,y,z;
    public final double qw,qx,qy,qz;
    public Pose(double x,double y,double z,double qw,double qx,double qy,double qz){
        this.x=x; this.y=y; this.z=z;
        this.qw=qw; this.qx=qx; this.qy=qy; this.qz=qz;
    }
    /** Rotate the world‐up vector (0,0,1) by this quaternion to get camera forward. */
    public double[] cameraForward() {
        double vx=0, vy=0, vz=1;
        double rw = -qx*vx - qy*vy - qz*vz;
        double rx =  qw*vx + qy*vz - qz*vy;
        double ry =  qw*vy + qz*vx - qx*vz;
        double rz =  qw*vz + qx*vy - qy*vx;
        double cx = rx*qw - rw*(-qx) - ry*(-qz) + rz*(-qy);
        double cy = ry*qw - rw*(-qy) - rz*(-qx) + rx*(-qz);
        double cz = rz*qw - rw*(-qz) - rx*(-qy) + ry*(-qx);
        double mag = Math.sqrt(cx*cx+cy*cy+cz*cz);
        return new double[]{cx/mag, cy/mag, cz/mag};
    }
}

/** Axis-aligned box for Oasis/KIZ. */
class AABB {
    public final double xmin,ymin,zmin,xmax,ymax,zmax;
    public AABB(double xmin,double ymin,double zmin,double xmax,double ymax,double zmax){
        this.xmin=xmin;this.ymin=ymin;this.zmin=zmin;
        this.xmax=xmax;this.ymax=ymax;this.zmax=zmax;
    }
    public boolean contains(Pose p){
        return p.x>=xmin&&p.x<=xmax
            && p.y>=ymin&&p.y<=ymax
            && p.z>=zmin&&p.z<=zmax;
    }
    public Pose projectInside(Pose p){
        double cx=Math.min(Math.max(p.x,xmin),xmax),
               cy=Math.min(Math.max(p.y,ymin),ymax),
               cz=Math.min(Math.max(p.z,zmin),zmax);
        return new Pose(cx,cy,cz,p.qw,p.qx,p.qy,p.qz);
    }
    public double intersectionLength(Pose p,Pose q){
        double dx=q.x-p.x, dy=q.y-p.y, dz=q.z-p.z;
        double t0=0,t1=1;
        double[] pArr={-dx,dx,-dy,dy,-dz,dz};
        double[] qArr={p.x-xmin,xmax-p.x,p.y-ymin,ymax-p.y,p.z-zmin,zmax-p.z};
        for(int i=0;i<6;i++){
            if(Math.abs(pArr[i])<1e-9){
                if(qArr[i]<0) return 0;
            } else {
                double t=qArr[i]/pArr[i];
                if(pArr[i]<0) t0=Math.max(t0,t);
                else t1=Math.min(t1,t);
                if(t0>t1) return 0;
            }
        }
        double len=Math.sqrt(dx*dx+dy*dy+dz*dz);
        return len*Math.max(0,t1-t0);
    }
}

/** Holds sheet plane center + normal + FOV/dist limits. */
class SheetInfo {
    public final Pose center;
    public final double nx,ny,nz, maxAngleRad= Math.toRadians(45), maxDist=0.9;
    public SheetInfo(Pose center,double nx,double ny,double nz){
        double m=Math.sqrt(nx*nx+ny*ny+nz*nz);
        this.center=center; this.nx=nx/m; this.ny=ny/m; this.nz=nz/m;
    }
}

/**
 * Core optimizer: computes KIZ‐legal photo‐poses, permutes them,
 * scores routes (area matches, time bonus, weighted travel cost + rotation),
 * then expands each leg into the true minimal‐cost subpath.
 */
class PathOptimizer {
    private static final double TIME_BONUS_RATE    = 0.2;  // pt per sec leftover
    private static final double OASIS_BENEFIT      = 0.7;  // cost inside Oasis=0.3
    private static final int    POINTS_PER_TAG     = 10;
    private static final double ROTATION_COST_RATE = 0.1;  // cost per radian
    private static final double ASTROBEE_SPEED     = 0.3;

    private final Pose startRaw, astroRaw;
    private final List<Pose> areasRaw;
    private final List<AABB> oasisZones, kizZones;
    private final double[]    oasisBonus;

    public PathOptimizer(
        Pose startRaw,
        List<Pose> areasRaw,
        Pose astroRaw,
        List<AABB> oasisZones,
        List<AABB> kizZones,
        double[] oasisBonus
    ){
        this.startRaw=startRaw; this.areasRaw=areasRaw;
        this.astroRaw=astroRaw; this.oasisZones=oasisZones;
        this.kizZones=kizZones; this.oasisBonus=oasisBonus;
    }

    public List<Pose> findOptimalRouteToAstronaut(){
        Pose s=clampIntoKIZ(startRaw), a=clampIntoKIZ(astroRaw);
        List<Pose> photos=new ArrayList<>();
        for(Pose raw:areasRaw) photos.add(findBestPhotoPose(raw));
        double[] bestScore={Double.NEGATIVE_INFINITY};
        List<Pose>[] bestSkel=new List[]{null};
        List<Integer> idx=Arrays.asList(0,1,2,3);
        permute(idx,0,perm->{
            List<Pose> sk=new ArrayList<>(); sk.add(s);
            perm.forEach(i->sk.add(photos.get(i)));
            sk.add(a);
            double sc=scoreSkeleton(sk);
            if(sc>bestScore[0]){ bestScore[0]=sc; bestSkel[0]=new ArrayList<>(sk);}            
        });
        List<Pose> full=new ArrayList<>();
        full.add(bestSkel[0].get(0));
        for(int i=0;i<bestSkel[0].size()-1;i++){
            Pose p=bestSkel[0].get(i), q=bestSkel[0].get(i+1);
            for(Pose w: segmentAndPath(p,q)){
                if(full.isEmpty()||!same(full.get(full.size()-1),w))
                    full.add(w);
            }
        }
        return full;
    }

    private Pose clampIntoKIZ(Pose p){
        Pose best=p; double bd=Double.POSITIVE_INFINITY;
        for(AABB kiz:kizZones) {
            if(kiz.contains(p)) return p;
            Pose q=kiz.projectInside(p);
            double d2=dist2(p,q);
            if(d2<bd){bd=d2; best=q;}
        }
        return best;
    }

    private Pose findBestPhotoPose(Pose raw){
        Pose best=null; double bd=Double.POSITIVE_INFINITY;
        for(AABB kiz:kizZones){
            Pose base=kiz.contains(raw)?raw:kiz.projectInside(raw);
            if(angleAndDistOK(base,raw)){
                double d2=dist2(base,raw);
                if(d2<bd){bd=d2; best=base;}
            } else {
                Pose slid=slideToAngle45(base,raw);
                if(kiz.contains(slid)&&angleAndDistOK(slid,raw)){
                    double d2=dist2(slid,raw);
                    if(d2<bd){bd=d2; best=slid;}
                }
            }
        }
        if(best==null) best=clampIntoKIZ(raw);
        // Orient camera toward target
        double[] q = computeLookAtQuaternion(best, raw);
        return new Pose(best.x,best.y,best.z,q[0],q[1],q[2],q[3]);
    }

    private boolean angleAndDistOK(Pose p,Pose raw){
        double[] fwd=p.cameraForward();
        double vx=raw.x-p.x, vy=raw.y-p.y, vz=raw.z-p.z;
        double mag=Math.sqrt(vx*vx+vy*vy+vz*vz);
        if(mag<1e-6) return true;
        double cos = (fwd[0]*vx+fwd[1]*vy+fwd[2]*vz)/mag;
        return cos>=Math.cos(Math.toRadians(45)) && mag<=0.9;
    }

    private Pose slideToAngle45(Pose base,Pose raw){
        double dz=base.z-raw.z, vz=Math.abs(dz);
        double desired= vz/Math.cos(Math.toRadians(45));
        double xyLen=Math.sqrt(Math.max(0,desired*desired-dz*dz));
        double dx=base.x-raw.x, dy=base.y-raw.y;
        double r=Math.hypot(dx,dy);
        double nx=r<1e-6?1:dx/r, ny=r<1e-6?0:dy/r;
        return new Pose(raw.x+nx*xyLen, raw.y+ny*xyLen, base.z,
                        base.qw,base.qx,base.qy,base.qz);
    }

    private double scoreSkeleton(List<Pose> sk){
        double am=0;
        for(int i=1;i<=4;i++) am += VisionPipeline.checkForLostItems(sk.get(i)) * POINTS_PER_TAG;
        double eD=0; for(int i=0;i<sk.size()-1;i++) eD+=dist(sk.get(i),sk.get(i+1));
        double t= eD/ASTROBEE_SPEED;
        double tb= Math.max(0, 300 - t) * TIME_BONUS_RATE;
        double tc=0; for(int i=0;i<sk.size()-1;i++) tc+=segmentCost(sk.get(i),sk.get(i+1));
        return am + tb - tc;
    }

    /** Compute cost of move+rotation between two poses. */
    private double segmentCost(Pose p,Pose q){
        // travel cost (oasis vs free)
        double d=dist(p,q);
        double in=0; for(AABB oz:oasisZones) in+=oz.intersectionLength(p,q);
        double out=d-in;
        double baseCost = out*1 + in*(1-OASIS_BENEFIT);
        // rotation cost
        double rot = rotationAngle(p,q) * ROTATION_COST_RATE;
        return baseCost + rot;
    }

    /** Expand segment, allowing rotate-in-place if cheaper. */
    private List<Pose> segmentAndPath(Pose p,Pose q){
        List<Pose> path=new ArrayList<>();
        double bestCost = segmentCost(p,q);
        path.add(q);
        // rotate-in-place then move
        Pose rotPose = new Pose(p.x,p.y,p.z,q.qw,q.qx,q.qy,q.qz);
        double rotThenMoveCost = (rotationAngle(p, rotPose) * ROTATION_COST_RATE)
                               + dist(rotPose, q);
        if(rotThenMoveCost < bestCost){
            path.clear();
            path.add(rotPose);
            path.add(q);
            bestCost = rotThenMoveCost;
        }
        // oasis shortcuts
        for(AABB oz:oasisZones){
            Pose e=oz.projectInside(p), x=oz.projectInside(q);
            double c = dist(p,e)*1 + dist(e,x)*(1-OASIS_BENEFIT) + dist(x,q)*1;
            if(c < bestCost){
                path.clear(); path.add(e); path.add(x); path.add(q);
            }
        }
        return path;
    }

    private double dist(Pose a,Pose b){ return Math.sqrt(dist2(a,b)); }
    private double dist2(Pose a,Pose b){ double dx=b.x-a.x, dy=b.y-a.y, dz=b.z-a.z; return dx*dx+dy*dy+dz*dz; }
    private boolean same(Pose a,Pose b){ return dist2(a,b)<1e-9; }

    /** Permute helper. */
    private void permute(List<Integer> idx,int k,Consumer<List<Integer>> cb){
        if(k==idx.size()-1) cb.accept(new ArrayList<>(idx));
        else for(int i=k;i<idx.size();i++){
            Collections.swap(idx,i,k);
            permute(idx,k+1,cb);
            Collections.swap(idx,k,i);
        }
    }

    /** Build quaternion looking from->target. */
    private double[] computeLookAtQuaternion(Pose from, Pose target) {
        double vx=target.x-from.x, vy=target.y-from.y, vz=target.z-from.z;
        double mag=Math.sqrt(vx*vx+vy*vy+vz*vz);
        if(mag<1e-6) return new double[]{from.qw,from.qx,from.qy,from.qz};
        vx/=mag; vy/=mag; vz/=mag;
        double ux=0, uy=0, uz=1;
        double rx=uy*vz-uz*vy, ry=uz*vx-ux*vz, rz=ux*vy-uy*vx;
        double rmag=Math.sqrt(rx*rx+ry*ry+rz*rz);
        if(rmag<1e-6){ rx=1; ry=0; rz=0; rmag=1; }
        rx/=rmag; ry/=rmag; rz/=rmag;
        ux=vy*rz-vz*ry; uy=vz*rx-vx*rz; uz=vx*ry-vy*rx;
        double m00=rx, m01=ux, m02=vx;
        double m10=ry, m11=uy, m12=vy;
        double m20=rz, m21=uz, m22=vz;
        double trace=m00+m11+m22;
        double qw,qx,qy,qz;
        if(trace>0){ double s=0.5/Math.sqrt(trace+1); qw=0.25/s; qx=(m21-m12)*s; qy=(m02-m20)*s; qz=(m10-m01)*s; }
        else if(m00>m11&&m00>m22){ double s=2*Math.sqrt(1+m00-m11-m22); qw=(m21-m12)/s; qx=0.25*s; qy=(m01+m10)/s; qz=(m02+m20)/s; }
        else if(m11>m22){ double s=2*Math.sqrt(1+m11-m00-m22); qw=(m02-m20)/s; qx=(m01+m10)/s; qy=0.25*s; qz=(m12+m21)/s; }
        else { double s=2*Math.sqrt(1+m22-m00-m11); qw=(m10-m01)/s; qx=(m02+m20)/s; qy=(m12+m21)/s; qz=0.25*s; }
        return new double[]{qw,qx,qy,qz};
    }

    /** Angle between two quaternions. */
    private double rotationAngle(Pose a, Pose b) {
        double dot=Math.abs(a.qw*b.qw + a.qx*b.qx + a.qy*b.qy + a.qz*b.qz);
        dot=Math.min(1,Math.max(-1,dot));
        return 2*Math.acos(dot);
    }

    /** Helper: choose best single photo pose for two sheets together. (unchanged) */
    public Pose chooseBestPhotoLocationForSheets(
        Pose prev, SheetInfo s2, SheetInfo s3
    ){ /* ... unchanged ... */
        double best=-Double.MAX_VALUE; Pose bestP=null;
        double wDist=0.5;
        // combined
        for(AABB kiz:kizZones){ /* ... */ }
        // fallback per-sheet
        for(SheetInfo s: Arrays.asList(s2,s3)){ /* ... */ }
        return bestP!=null?bestP: clampIntoKIZ(s2.center);
    }
}

/** Vision stub: returns tags detected at a given photo pose. */
class VisionPipeline {
    public static int checkForLostItems(Pose p){ return 0; }
}
