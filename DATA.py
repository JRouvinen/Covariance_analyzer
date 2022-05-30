class DATA:
    def __init__(self, time_tag,lat_lon_xx, lat_lon_xy, lat_lon_yy, alt_xz,
                 alt_yz, alt_zz, vel_xx, vel_xy, vel_yy, hdg):

        self.time_tag = time_tag
        self.lat_lon_xx = lat_lon_xx
        self.lat_lon_xy = lat_lon_xy
        self.lat_lon_yy = lat_lon_yy
        self.alt_xz = alt_xz
        self.alt_yz = alt_yz
        self.alt_zz = alt_zz
        self.vel_xx = vel_xx
        self.vel_xy = vel_xy
        self.vel_yy = vel_yy
        self.hdg = hdg

    def getData(self):
        covariance_dict = {
            "time_tag":self.time_tag,
            "lat_lon_xx": self.lat_lon_xx,
            "lat_lon_xy": self.lat_lon_xy,
            "lat_lon_yy": self.lat_lon_yy,
            "alt_xz": self.alt_xz,
            "alt_yz": self.alt_yz,
            "alt_zz": self.alt_zz,
            "vel_xx": self.vel_xx,
            "vel_xy": self.vel_xy,
            "vel_yy": self.vel_yy,
            "hdg": self.hdg,
        }
        return covariance_dict