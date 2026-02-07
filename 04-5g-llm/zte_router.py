"""
ZTE Router API Client - Python port of ZTE.js
Original code by Miononno, enhanced by lteforum.at
Python conversion with Ollama integration
"""

import hashlib
import requests
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LteCellInfo:
    """LTE cell signal information"""
    pci: int = 0
    band: str = ""
    earfcn: str = ""
    bandwidth: str = ""
    rssi: str = ""
    rsrp1: str = ""
    rsrp2: str = ""
    rsrp3: str = ""
    rsrp4: str = ""
    rsrq: str = ""
    sinr1: str = ""
    sinr2: str = ""
    sinr3: str = ""
    sinr4: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pci": self.pci, "band": self.band, "earfcn": self.earfcn,
            "bandwidth": self.bandwidth, "rssi": self.rssi,
            "rsrp1": self.rsrp1, "rsrp2": self.rsrp2, "rsrp3": self.rsrp3, "rsrp4": self.rsrp4,
            "rsrq": self.rsrq,
            "sinr1": self.sinr1, "sinr2": self.sinr2, "sinr3": self.sinr3, "sinr4": self.sinr4
        }


@dataclass
class NrCellInfo:
    """5G NR cell signal information"""
    pci: int = 0
    band: str = ""
    arfcn: str = ""
    bandwidth: str = ""
    rsrp1: str = ""
    rsrp2: str = ""
    rsrq: str = ""
    sinr: str = ""
    info_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pci": self.pci, "band": self.band, "arfcn": self.arfcn,
            "bandwidth": self.bandwidth,
            "rsrp1": self.rsrp1, "rsrp2": self.rsrp2, "rsrq": self.rsrq,
            "sinr": self.sinr, "info_text": self.info_text
        }


class ZTERouter:
    """ZTE Router API client for signal monitoring and configuration"""

    # Signal info parameters to request from router
    SIGINFO_PARAMS = (
        "cell_id,dns_mode,prefer_dns_manual,standby_dns_manual,network_type,net_select,BearerPreference,"
        "network_provider_fullname,rmcc,rmnc,ip_passthrough_enabled,bandwidth,tx_power,ppp_status,"
        "rscp_1,ecio_1,rscp_2,ecio_2,rscp_3,ecio_3,rscp_4,ecio_4,"
        "ngbr_cell_info,lte_multi_ca_scell_info,lte_multi_ca_scell_sig_info,"
        "lte_band,lte_rsrp,lte_rsrq,lte_rssi,lte_snr,"
        "lte_ca_pcell_band,lte_ca_pcell_freq,lte_ca_pcell_bandwidth,"
        "lte_ca_scell_band,lte_ca_scell_bandwidth,"
        "lte_rsrp_1,lte_rsrp_2,lte_rsrp_3,lte_rsrp_4,"
        "lte_snr_1,lte_snr_2,lte_snr_3,lte_snr_4,"
        "lte_pci,lte_pci_lock,lte_earfcn_lock,"
        "5g_rx0_rsrp,5g_rx1_rsrp,Z5g_rsrp,Z5g_rsrq,Z5g_SINR,"
        "nr5g_cell_id,nr5g_pci,nr5g_action_channel,nr5g_action_band,nr5g_action_nsa_band,"
        "nr_ca_pcell_band,nr_ca_pcell_freq,nr_multi_ca_scell_info,"
        "nr5g_sa_band_lock,nr5g_nsa_band_lock,"
        "pm_sensor_ambient,pm_sensor_mdm,pm_sensor_5g,pm_sensor_pa1,wifi_chip_temp"
    )

    # Network mode options
    NETWORK_MODES = [
        "Only_GSM", "Only_WCDMA", "Only_LTE", "WCDMA_AND_GSM", "WCDMA_preferred",
        "WCDMA_AND_LTE", "GSM_AND_LTE", "CDMA_EVDO_LTE", "Only_TDSCDMA",
        "TDSCDMA_AND_WCDMA", "TDSCDMA_AND_LTE", "TDSCDMA_WCDMA_HDR_CDMA_GSM_LTE",
        "TDSCDMA_WCDMA_GSM_LTE", "GSM_WCDMA_LTE", "Only_5G", "LTE_AND_5G",
        "GWL_5G", "TCHGWL_5G", "WL_AND_5G", "TGWL_AND_5G", "4G_AND_5G"
    ]

    def __init__(self, ip_address: str):
        """Initialize router connection"""
        self.base_url = f"http://{ip_address}"
        self.session = requests.Session()
        # Required headers for ZTE routers
        self.session.headers.update({
            "Referer": f"http://{ip_address}/index.html",
            "Origin": f"http://{ip_address}",
            "X-Requested-With": "XMLHttpRequest",
        })
        self.is_logged_in = False
        self.is_mc888 = False
        self.is_mc889 = False
        self.is_mc801 = False
        self._signal_data: Dict[str, Any] = {}
        self._password: Optional[str] = None
        self._password_hash: Optional[str] = None

    def _sha256(self, text: str) -> str:
        """Calculate SHA256 hash (uppercase)"""
        return hashlib.sha256(text.encode()).hexdigest().upper()

    def _md5(self, text: str) -> str:
        """Calculate MD5 hash (uppercase)"""
        return hashlib.md5(text.encode()).hexdigest().upper()

    def _hash(self, text: str) -> str:
        """Use appropriate hash based on device model"""
        if self.is_mc888 or self.is_mc889 or self.is_mc801:
            return self._sha256(text)
        return self._md5(text)

    def _get_cmd(self, cmd: str, multi_data: bool = False) -> Dict[str, Any]:
        """Execute GET command on router"""
        params = {"cmd": cmd}
        if multi_data:
            params["multi_data"] = "1"
        try:
            resp = self.session.get(
                f"{self.base_url}/goform/goform_get_cmd_process",
                params=params, timeout=5
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _set_cmd(self, data: Dict[str, Any], retry_on_failure: bool = True) -> Dict[str, Any]:
        """Execute POST command on router with optional auto-relogin on failure"""
        try:
            resp = self.session.post(
                f"{self.base_url}/goform/goform_set_cmd_process",
                data=data, timeout=10
            )
            result = resp.json()
            
            # If failed and we have credentials, try relogin once
            if result.get("result") == "failure" and retry_on_failure and self._password:
                print("[DEBUG] Command failed, attempting auto-relogin...")
                login_res = self.login(self._password)
                if login_res.get("success"):
                    print("[DEBUG] Relogin successful, retrying command...")
                    # Update AD if present in original data
                    if "AD" in data:
                        wa_inner, cr_version, rd, _ = self._get_auth_data()
                        data["AD"] = self._compute_ad(wa_inner, cr_version, rd)
                    return self._set_cmd(data, retry_on_failure=False)
            
            print(f"[DEBUG] SET {data.get('goformId', 'CMD')} result: {result}")
            if result.get("result") == "failure":
                print(f"[DEBUG] FULL DATA SENT: {data}")
            return result
        except Exception as e:
            return {"error": str(e)}

    def _is_success(self, result: Dict[str, Any]) -> bool:
        """Centralized success check for router commands"""
        if not result:
            return False
        res = str(result.get("result", "")).lower()
        return res in ["0", "success", "ok"]

    def _get_auth_data(self) -> tuple:
        """Get authentication data (wa_inner_version, cr_version, RD, LD)"""
        data = self._get_cmd("wa_inner_version,cr_version,RD,LD", multi_data=True)
        return (
            data.get("wa_inner_version", ""),
            data.get("cr_version", ""),
            data.get("RD", ""),
            data.get("LD", "")
        )

    def _compute_ad(self, wa_inner: str, cr_version: str, rd: str) -> str:
        """Compute AD authentication token"""
        return self._hash(self._hash(wa_inner + cr_version) + rd)

    def detect_model(self) -> bool:
        """Detect router model (MC888, MC889, MC801, etc)"""
        data = self._get_cmd("wa_inner_version")
        version = data.get("wa_inner_version", "")
        self.is_mc888 = "MC888" in version
        self.is_mc889 = "MC889" in version
        self.is_mc801 = "MC801" in version or "801" in version
        print(f"[DEBUG] Detected model: MC888={self.is_mc888}, MC889={self.is_mc889}, MC801={self.is_mc801}")
        return bool(version)

    def check_login(self) -> bool:
        """Check if currently logged in"""
        data = self._get_cmd("loginfo", multi_data=True)
        return data.get("loginfo", "").lower() == "ok"

    def login(self, password: str) -> Dict[str, Any]:
        """Login to router with password"""
        self._password = password
        # Detect model first
        if not self.detect_model():
            return {"success": False, "error": "Could not connect to router. Check IP address."}

        # Get auth data
        wa_inner, cr_version, rd, ld = self._get_auth_data()
        print(f"[DEBUG] Auth Data: wa_inner='{wa_inner}', cr_version='{cr_version}', rd='{rd}', ld='{ld}'")

        if not wa_inner:
            return {"success": False, "error": "Could not get router version info"}

        # Compute password hash and AD token
        # Most ZTE Routers use SHA256 for the base password hash
        self._password_hash = self._sha256(password)
        
        # AD token usually uses self._hash (MD5 for older, SHA256 for newer)
        ad = self._compute_ad(wa_inner, cr_version, rd)
        
        # Second hash for password token always uses SHA256 based on password_hash + ld
        password_token = self._sha256(self._password_hash + ld)
        
        print(f"[DEBUG] AD: {ad}, Len: {len(ad)}")
        print(f"[DEBUG] Password Token: {password_token[:10]}..., Len: {len(password_token)}")

        # Perform login
        result = self._set_cmd({
            "isTest": "false",
            "goformId": "LOGIN",
            "password": password_token,
            "AD": ad
        })
        print(f"[DEBUG] Raw login response: {result}")

        # Check for connection errors first
        if "error" in result:
            return {"success": False, "error": f"Connection error: {result['error']}"}

        if result.get("result") == "0":
            self.is_logged_in = True
            return {"success": True}
        else:
            error_map = {
                "1": "Router is busy or locked out. Please wait 5-10 minutes and make sure you are logged out of the router's web interface (192.168.0.1).",
                "3": "Wrong password. Double check your router admin password.",
                "failure": "Router rejected connection. Another session might be active."
            }
            res_code = result.get("result", "")
            error = error_map.get(res_code, f"Login failed (code: {res_code})")
            return {"success": False, "error": error}

    def get_signal_info(self) -> Dict[str, Any]:
        """Get comprehensive signal information"""
        data = self._get_cmd(self.SIGINFO_PARAMS, multi_data=True)
        if "error" in data:
            return data

        self._signal_data = data

        # Determine network type
        network_type = data.get("network_type", "")
        is_umts = network_type in ["HSPA", "HSDPA", "HSUPA", "HSPA+", "DC-HSPA+",
                                   "UMTS", "CDMA", "CDMA_EVDO", "EVDO_EHRPD", "TDSCDMA"]
        is_lte = network_type in ["LTE", "ENDC", "EN-DC", "LTE-NSA"]
        is_lte_plus = data.get("wan_lte_ca") in ["ca_activated", "ca_deactivated"]
        is_5g_sa = network_type == "SA"
        is_5g_nsa = network_type in ["ENDC", "EN-DC", "LTE-NSA"]
        is_5g_nsa_active = is_5g_nsa and network_type != "LTE-NSA"
        is_5g = is_5g_sa or is_5g_nsa

        # Parse LTE cells
        lte_cells = self._parse_lte_cells(data) if is_lte else []

        # Parse NR (5G) cells
        nr_cells = self._parse_nr_cells(data) if is_5g else []

        # Parse temperature
        temps = {}
        for sensor, label in [
            ("pm_sensor_ambient", "ambient"),
            ("pm_sensor_mdm", "modem"),
            ("pm_sensor_5g", "5g"),
            ("pm_sensor_pa1", "pa"),
            ("wifi_chip_temp", "wifi")
        ]:
            val = data.get(sensor, "")
            if val and float(val) > -40:
                temps[label] = float(val)

        # Get band info string
        lte_bands = self._get_band_info(lte_cells)
        nr_bands = self._get_band_info(nr_cells)
        band_info = lte_bands
        if nr_bands:
            band_info += (" + " if band_info else "") + nr_bands

        return {
            "network_type": network_type,
            "is_umts": is_umts,
            "is_lte": is_lte,
            "is_lte_plus": is_lte_plus,
            "is_5g": is_5g,
            "is_5g_sa": is_5g_sa,
            "is_5g_nsa": is_5g_nsa,
            "is_5g_nsa_active": is_5g_nsa_active,
            "lte_cells": [c.to_dict() for c in lte_cells],
            "nr_cells": [c.to_dict() for c in nr_cells],
            "band_info": band_info,
            "provider": data.get("network_provider_fullname", ""),
            "cell_id": data.get("cell_id", ""),
            "nr_cell_id": data.get("nr5g_cell_id", ""),
            "wan_ip": data.get("wan_ipaddr", ""),
            "tx_power": data.get("tx_power", ""),
            "ca_active": data.get("wan_lte_ca") == "ca_activated",
            "temperatures": temps,
            "lte_pci_lock": data.get("lte_pci_lock", "0"),
            "lte_earfcn_lock": data.get("lte_earfcn_lock", "0"),
            "nr5g_sa_band_lock": data.get("nr5g_sa_band_lock", ""),
            "nr5g_nsa_band_lock": data.get("nr5g_nsa_band_lock", ""),
            # UMTS data
            "umts": {
                "rscp_1": data.get("rscp_1", ""),
                "ecio_1": data.get("ecio_1", ""),
                "rscp_2": data.get("rscp_2", ""),
                "ecio_2": data.get("ecio_2", ""),
                "rscp_3": data.get("rscp_3", ""),
                "ecio_3": data.get("ecio_3", ""),
                "rscp_4": data.get("rscp_4", ""),
                "ecio_4": data.get("ecio_4", ""),
            } if is_umts else None
        }

    def _parse_lte_cells(self, data: Dict) -> List[LteCellInfo]:
        """Parse LTE cell information"""
        cells = []

        # Primary cell
        lte_main_band = data.get("lte_ca_pcell_band") or data.get("lte_band") or "??"
        bandwidth = (data.get("lte_ca_pcell_bandwidth") or data.get("bandwidth", ""))
        bandwidth = bandwidth.replace("MHz", "").replace(".0", "")

        cells.append(LteCellInfo(
            pci=int(data.get("lte_pci", "0"), 16) if data.get("lte_pci") else 0,
            band=f"B{lte_main_band}",
            earfcn=data.get("lte_ca_pcell_freq") or data.get("wan_active_channel", ""),
            bandwidth=bandwidth,
            rssi=data.get("lte_rssi", ""),
            rsrp1=data.get("lte_rsrp_1", ""),
            rsrp2=data.get("lte_rsrp_2", ""),
            rsrp3=data.get("lte_rsrp_3", ""),
            rsrp4=data.get("lte_rsrp_4", ""),
            rsrq=data.get("lte_rsrq", ""),
            sinr1=data.get("lte_snr_1", ""),
            sinr2=data.get("lte_snr_2", ""),
            sinr3=data.get("lte_snr_3", ""),
            sinr4=data.get("lte_snr_4", "")
        ))

        # Secondary cells (CA)
        scell_infos = [s for s in data.get("lte_multi_ca_scell_info", "").split(";") if s]
        scell_sig_infos = [s for s in data.get("lte_multi_ca_scell_sig_info", "").split(";") if s]

        for i, scell_info_str in enumerate(scell_infos):
            parts = scell_info_str.split(",")
            if len(parts) < 6:
                continue

            sig_parts = scell_sig_infos[i].split(",") if i < len(scell_sig_infos) else []
            has_sig = len(sig_parts) >= 3

            cells.append(LteCellInfo(
                pci=int(parts[1], 16) if parts[1] != "XX" else 0,
                band=f"B{parts[3]}",
                earfcn=parts[4],
                bandwidth=parts[5].replace(".0", ""),
                rsrp1=sig_parts[0].replace("-44.0", "?????") if has_sig else "",
                rsrq=sig_parts[1] if has_sig else "",
                sinr1=sig_parts[2] if has_sig else ""
            ))

        return cells

    def _parse_nr_cells(self, data: Dict) -> List[NrCellInfo]:
        """Parse 5G NR cell information"""
        cells = []

        network_type = data.get("network_type", "")
        is_5g_nsa = network_type in ["ENDC", "EN-DC", "LTE-NSA"]
        is_5g_nsa_active = is_5g_nsa and network_type != "LTE-NSA"

        if is_5g_nsa and not is_5g_nsa_active:
            return []

        rsrp1 = data.get("5g_rx0_rsrp") or data.get("Z5g_rsrp", "")
        rsrp2 = data.get("5g_rx1_rsrp", "")
        sinr = (data.get("Z5g_SINR", "")
                .replace("-20.0", "?????")
                .replace("-3276.8", "?????"))

        nr_band = data.get("nr5g_action_nsa_band") if is_5g_nsa else data.get("nr5g_action_band", "")
        if not nr_band or nr_band == "-1":
            nr_band = data.get("nr_ca_pcell_band", "??")

        # Ensure band starts with 'n'
        if nr_band and not nr_band.startswith("n"):
            nr_band = f"n{nr_band}"

        cells.append(NrCellInfo(
            pci=int(data.get("nr5g_pci", "0"), 16) if data.get("nr5g_pci") else 0,
            band=nr_band,
            arfcn=data.get("nr_ca_pcell_freq") or data.get("nr5g_action_channel", ""),
            bandwidth=data.get("bandwidth", "").replace("MHz", "") if not is_5g_nsa else "",
            rsrp1=rsrp1,
            rsrp2=rsrp2,
            rsrq=data.get("Z5g_rsrq", ""),
            sinr=sinr
        ))

        # Secondary cells (CA)
        for scell_str in data.get("nr_multi_ca_scell_info", "").split(";"):
            if not scell_str:
                continue
            parts = scell_str.split(",")
            if len(parts) < 10:
                continue

            cells.append(NrCellInfo(
                pci=int(parts[1]) if parts[1].isdigit() else 0,
                band=parts[3],
                arfcn=parts[4],
                bandwidth=parts[5].replace("MHz", ""),
                rsrp1=parts[7],
                rsrq=parts[8],
                sinr=parts[9].replace("0.0", "?????")
            ))

        return cells

    def _get_band_info(self, cells) -> str:
        """Get band info string from cell list"""
        bands = []
        for cell in cells:
            info = cell.band
            if cell.bandwidth:
                info += f"({cell.bandwidth}MHz)"
            bands.append(info)
        return " + ".join(bands)

    def set_network_mode(self, mode: str) -> Dict[str, Any]:
        """Set network mode (Only_LTE, WL_AND_5G, etc)"""
        if mode not in self.NETWORK_MODES:
            return {"success": False, "error": f"Invalid mode. Valid: {', '.join(self.NETWORK_MODES)}"}

        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        # Mapping for different ZTE firmware versions
        mode_map = {
            "Only_LTE": "4G_Only",
            "Only_5G": "5G_Only",
            "LTE_AND_5G": "LTE_AND_5G",
            "WL_AND_5G": "Automatic"
        }
        net_select_val = mode_map.get(mode, mode)

        # Try multiple variants for the mode string
        variants = [net_select_val]
        if mode == "Only_LTE": variants.extend(["4G_Only", "LTE_Only", "Only_LTE"])
        if mode == "Only_5G": variants.extend(["5G_Only", "NR5G_Only", "Only_5G"])
        if mode == "WL_AND_5G": variants.extend(["Automatic", "WL_AND_5G"])
        
        # Unique variants only
        variants = list(dict.fromkeys(variants))

        for variant in variants:
            print(f"[DEBUG] Attempting mode change with variant: {variant} using NET_SELECT")
            data = {
                "isTest": "false",
                "goformId": "NET_SELECT",
                "net_select": variant,
                "AD": ad
            }
            result = self._set_cmd(data)
            
            if not self._is_success(result):
                print(f"[DEBUG] NET_SELECT ({variant}) failed, trying with disconnect toggle...")
                self.disconnect_network()
                time.sleep(3) # Longer wait for MC801A
                result = self._set_cmd(data)
                time.sleep(2)
                self.connect_network()
                time.sleep(1)

            if self._is_success(result):
                return {"success": True, "result": result, "applied_variant": variant}

        # Final Fallback to older SET_BEARER_PREFERENCE
        print(f"[DEBUG] All NET_SELECT variants failed. Falling back to SET_BEARER_PREFERENCE...")
        result = self._set_cmd({
            "isTest": "false",
            "goformId": "SET_BEARER_PREFERENCE",
            "BearerPreference": mode,
            "AD": ad
        })

        return {"success": self._is_success(result), "result": result}

    def set_lte_bands(self, bands: str) -> Dict[str, Any]:
        """
        Set LTE bands
        bands: "AUTO" or bands separated by + (e.g., "1+3+20")
        """
        if bands.upper() == "AUTO":
            band_mask = "0xA3E2AB0908DF"
        else:
            band_list = bands.split("+")
            mask = sum(pow(2, int(b) - 1) for b in band_list)
            band_mask = f"0x{mask:011X}"

        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "BAND_SELECT",
            "is_gw_band": "0",
            "gw_band_mask": "0",
            "is_lte_band": "1",
            "lte_band_mask": band_mask,
            "AD": ad
        })

        # If it failed, try disconnecting first
        if not self._is_success(result):
            print("[DEBUG] LTE Band change rejected, trying with disconnect toggle...")
            self.disconnect_network()
            import time
            time.sleep(1)
            result = self._set_cmd({
                "isTest": "false",
                "goformId": "BAND_SELECT",
                "is_gw_band": "0",
                "gw_band_mask": "0",
                "is_lte_band": "1",
                "lte_band_mask": band_mask,
                "AD": ad
            })
            time.sleep(1)
            self.connect_network()

        return {"success": self._is_success(result), "result": result}

    def set_nr_bands(self, bands: str) -> Dict[str, Any]:
        """
        Set 5G NR bands
        bands: "AUTO" or bands separated by + (e.g., "1+78")
        """
        if bands.upper() == "AUTO":
            band_mask = "1,2,3,5,7,8,20,28,38,41,50,51,66,70,71,74,75,76,77,78,79,80,81,82,83,84"
        else:
            band_mask = bands.replace("+", ",")

        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "WAN_PERFORM_NR5G_BAND_LOCK",
            "nr5g_band_mask": band_mask,
            "AD": ad
        })

        # If it failed, try disconnecting first
        if not self._is_success(result):
            print("[DEBUG] NR Band change rejected, trying with disconnect toggle...")
            self.disconnect_network()
            import time
            time.sleep(1)
            result = self._set_cmd({
                "isTest": "false",
                "goformId": "WAN_PERFORM_NR5G_BAND_LOCK",
                "nr5g_band_mask": band_mask,
                "AD": ad
            })
            time.sleep(1)
            self.connect_network()

        return {"success": self._is_success(result), "result": result}

    def lock_lte_cell(self, pci: int = 0, earfcn: int = 0) -> Dict[str, Any]:
        """Lock to specific LTE cell. Pass 0,0 to reset."""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "LTE_LOCK_CELL_SET",
            "lte_pci_lock": str(pci),
            "lte_earfcn_lock": str(earfcn),
            "AD": ad
        })

        return {"success": self._is_success(result), "result": result}

    def lock_nr_cell(self, pci: int = 0, arfcn: int = 0, band: int = 0, scs: int = 30) -> Dict[str, Any]:
        """Lock to specific 5G NR cell. Pass 0,0,0,0 to reset."""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "NR5G_LOCK_CELL_SET",
            "nr5g_cell_lock": f"{pci},{arfcn},{band},{scs}",
            "AD": ad
        })

        return {"success": self._is_success(result), "result": result}

    def set_bridge_mode(self, enable: bool) -> Dict[str, Any]:
        """Enable or disable bridge mode"""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "OPERATION_MODE",
            "opMode": "LTE_BRIDGE" if enable else "PPP",
            "ethernet_port_specified": "1",
            "AD": ad
        })

        return {"success": self._is_success(result), "result": result}

    def set_arp_proxy(self, enable: bool) -> Dict[str, Any]:
        """Enable or disable ARP proxy"""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "ARP_PROXY_SWITCH",
            "arp_proxy_switch": "1" if enable else "0",
            "AD": ad
        })

        return {"success": self._is_success(result), "result": result}

    def reboot(self) -> Dict[str, Any]:
        """Reboot the router"""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)

        result = self._set_cmd({
            "isTest": "false",
            "goformId": "REBOOT_DEVICE",
            "AD": ad
        })

        return {"success": True, "result": result}

    def connect_network(self) -> Dict[str, Any]:
        """Connect to mobile network"""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)
        return self._set_cmd({"isTest": "false", "goformId": "CONNECT_NETWORK", "AD": ad})

    def disconnect_network(self) -> Dict[str, Any]:
        """Disconnect from mobile network"""
        wa_inner, cr_version, rd, _ = self._get_auth_data()
        ad = self._compute_ad(wa_inner, cr_version, rd)
        return self._set_cmd({"isTest": "false", "goformId": "DISCONNECT_NETWORK", "AD": ad})

    def get_version_info(self) -> Dict[str, Any]:
        """Get router version information"""
        data = self._get_cmd("hardware_version,web_version,wa_inner_version,cr_version", multi_data=True)
        return {
            "hardware": data.get("hardware_version", ""),
            "web": data.get("web_version", ""),
            "wa_inner": data.get("wa_inner_version", ""),
            "cr": data.get("cr_version", "")
        }
