#!/usr/bin/env python
#
# Hi There!
#
# You may be wondering what this giant blob of binary data here is, you might
# even be worried that we're up to something nefarious (good for you for being
# paranoid!). This is a base85 encoding of a zip file, this zip file contains
# an entire copy of pip (version 26.1.1).
#
# Pip is a thing that installs packages, pip itself is a package that someone
# might want to install, especially if they're looking to run this get-pip.py
# script. Pip has a lot of code to deal with the security of installing
# packages, various edge cases on various platforms, and other such sort of
# "tribal knowledge" that has been encoded in its code base. Because of this
# we basically include an entire copy of pip inside this blob. We do this
# because the alternatives are attempt to implement a "minipip" that probably
# doesn't do things correctly and has weird edge cases, or compress pip itself
# down into a single file.
#
# If you're wondering how this is created, it is generated using
# `scripts/generate.py` in https://github.com/pypa/get-pip.

import sys

this_python = sys.version_info[:2]
min_version = (3, 10)
if this_python < min_version:
    message_parts = [
        "This script does not work on Python {}.{}.".format(*this_python),
        "The minimum supported Python version is {}.{}.".format(*min_version),
        "Please use https://bootstrap.pypa.io/pip/{}.{}/get-pip.py instead.".format(*this_python),
    ]
    print("ERROR: " + " ".join(message_parts))
    sys.exit(1)


import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode


def include_setuptools(args):
    """
    Install setuptools only if absent, not excluded and when using Python <3.12.
    """
    cli = not args.no_setuptools
    env = not os.environ.get("PIP_NO_SETUPTOOLS")
    absent = not importlib.util.find_spec("setuptools")
    python_lt_3_12 = this_python < (3, 12)
    return cli and env and absent and python_lt_3_12


def include_wheel(args):
    """
    Install wheel only if absent, not excluded and when using Python <3.12.
    """
    cli = not args.no_wheel
    env = not os.environ.get("PIP_NO_WHEEL")
    absent = not importlib.util.find_spec("wheel")
    python_lt_3_12 = this_python < (3, 12)
    return cli and env and absent and python_lt_3_12


def determine_pip_install_arguments():
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("--no-setuptools", action="store_true")
    pre_parser.add_argument("--no-wheel", action="store_true")
    pre, args = pre_parser.parse_known_args()

    args.append("pip")

    if include_setuptools(pre):
        args.append("setuptools")

    if include_wheel(pre):
        args.append("wheel")

    return ["install", "--upgrade", "--force-reinstall"] + args


def monkeypatch_for_cert(tmpdir):
    """Patches `pip install` to provide default certificate with the lowest priority.

    This ensures that the bundled certificates are used unless the user specifies a
    custom cert via any of pip's option passing mechanisms (config, env-var, CLI).

    A monkeypatch is the easiest way to achieve this, without messing too much with
    the rest of pip's internals.
    """
    from pip._internal.commands.install import InstallCommand

    # We want to be using the internal certificates.
    cert_path = os.path.join(tmpdir, "cacert.pem")
    with open(cert_path, "wb") as cert:
        cert.write(pkgutil.get_data("pip._vendor.certifi", "cacert.pem"))

    install_parse_args = InstallCommand.parse_args

    def cert_parse_args(self, args):
        if not self.parser.get_default_values().cert:
            # There are no user provided cert -- force use of bundled cert
            self.parser.defaults["cert"] = cert_path  # calculated above
        return install_parse_args(self, args)

    InstallCommand.parse_args = cert_parse_args


def bootstrap(tmpdir):
    monkeypatch_for_cert(tmpdir)

    # Execute the included pip and use it to install the latest pip and
    # any user-requested packages from PyPI.
    from pip._internal.cli.main import main as pip_entry_point
    args = determine_pip_install_arguments()
    sys.exit(pip_entry_point(args))


def main():
    tmpdir = None
    try:
        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()

        # Unpack the zipfile into the temporary directory
        pip_zip = os.path.join(tmpdir, "pip.zip")
        with open(pip_zip, "wb") as fp:
            fp.write(b85decode(DATA.replace(b"\n", b"")))

        # Add the zipfile to sys.path so that we can import it
        sys.path.insert(0, pip_zip)

        # Run the bootstrap
        bootstrap(tmpdir=tmpdir)
    finally:
        # Clean up our temporary working directory
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


DATA = b"""
P)h>@6aWAK2mk|@q+Bf6Ax8QD003hF000jF003}la4%n9X>MtBUtcb8c|DLpOT<77h41q#LNB_YQ&vR
R1qCmH7xCatWSDK!-GMeUB&kcmA8%TzVIafH<MHu2&I5$djXE-h0BI<h6(UjAs40^;7s5BP*x&AtP~F
`5t>0G8xfVMZVxr5)N7+N4L_bCO3x41&6PkHm8@PUgM7noiQ&rW+DGAt%G|R{odw70-g-rbf14dHlGQ
%hchY3n57XtZA&=^hb5v1W%RJ>aPU(6cYHqEdW)S|}J%M}PBoK%bK>-w1VG#-4Dhq_E9)|Oc(krAc7z
kS&Gm2BDFT!}e+Sn2$z7U_|cr;<&TwWo0ASPJqV3Zu11|Kd{1#{B|NO9KQH0000800Wn#T;<AP#l!&s
0BQpO01p5F0B~t=FJE76VQFq(UoLQYT~bYtn=lZ)^D9Q#15mMS&Q-Hgk9%G1t*R``48aQ6mdAwBU*8!
AlHGDL_<hYA>=b-V;Hj{;6RlJfPw<YDzao?+XxD+6V?>u=@$q8a89E!$Bp+Akqn{uR2)&JzfI)F(y6;
7(4LS`C*d9Ve5`pAFU%l;MCehh-?|MwN4uTC}{4}vOff>+T8a6`wi!A0S>2YjewCpG0Xz)wJ#AQIo*H
?eR4m3en)8HLEPg(EBCiq&|N48(b-{Myt4h>h(o-vuFtLplD0sQQ!Huz8-YpLa}hXp}Lgg84VBjf_Km
?MYMU6<R)uQKhf!Q79*5^!|CP{z!p&$y@Lh|k~JWf)&1>^9`%tAutwfl?IEC<Q*=&#a$IJe}ZhxFvt$
%ifO2K=%Bi{i4*QtrOb1VQ;JB9XLH97%`?4Z6H}=`I%5QKxn*_z5@iovgM}rjx>)+ijk?sT}I8qEcU3
zCc?4bNI!FHI(;q-$V=5m^G<JnCc9*OzCFpl@^ZG^)&K8%19km%ncOYLR2`cv#9QTRl@bcKB^~bbZe8
XDvv;ekOUT+|Yew4&W>1#S{KsWtaks-iHdXl>|5C+2P)h>@6aWAK2mk|@q+H@elhf7$00656000#L00
3}la4%n9aA|NYa&>NQWpZC%E^v8$RKZT$KoGtAD+Y6@Eg6?WPf=4<LP6z1(j*)zidGwYOqSZaW_L|vA
^yEHcAP*&>qD~Moq6--&0B|IIN4y<cm&+Y%$F_rwQ`V|x&>1}t?C5+DB9B?Ay-9#7PSY|Ps$oy(K-!+
C8rdsM4*2yE6hPUP@Y~B@P?vYW08=h@8K(xHx!*o{lUYE(yH26Piap|22yX|!FlPiuMvh}5FEqDqhHW
fdqV4!(L}VWrB^FeD&Io3#_TsZ%sih}jqA{7tzXmY*)F)ml#zm9sM9(Izc95T4uv%p)NO6e0|BMpRAn
&xx%@u87W2uEo2&U`0b_{H6Zmu*2SJcyG0+jt+i1IvM*zp*+Zsop?zOQP06i9cV>N_nm_8V#qb?(c!d
Q3$QhCTGU{x@)bih`+Ft?+#JNN<zbVfb0R1RU$I?`;iQ9kN+`3c`h){hl(gETBd-8XfoLfy;s`~s21$
3|H?|2lqtiHR!FtK=(0eqQ(d9?v*=6LP>g-eL8tm?v<_(yz&4U^Mxza|kqUH^Or$P?CN~zFIYGCM1cp
mz#dr7AvVV0lhc!-MDR|$=x&@Z64(aCZA8wf^O*jd8t-uB{Q--+IFzwO#1bu(mC4q5{*l#%6bUSmE~J
-5E;y_W@0hDnO{s6Vm_TN$wk}9X!sX`uaOWj&p#`rb1+l&RR=pqiehW)`<_WZ_HSKSh177~ZLl8;3J|
WwEOJ;D2^39Pa)AVx=lI%aT6$^k(RVNZFe^Xg!XX=u#t@Qs!qFQeY%R?=!R~;IQ-gcm(mkg=o7+Q8?g
m<We^7VGmy|LShz=9>z3nr$ggAp_WN(r@QgG|3(u@|JCjSvOV)?gag<xzUU`)LW!Tu5=Od!0Fnr{!8i
DF2U6IzjDgh<I;^!bxCdws(3+s|DXbjC~RSM)nkCuYJJU6t|c0<!dkM&!p!+fZ|V0Z>Z=1QY-O00;mB
m!w?H$G|nf0000U0RR9D0001RX>c!ac`kH$aAjmAj<HU~FbqZae#PBbp85|A3~X;eVm7U5EhTo8IEDN
@P8ls7-*bu-NCRQBoJn^iQAVkDRLUzpPe}~%$w)4VGpq9sQ9OsArVq@gW&td8ktF(xhi|JBx9SfJ>&U
%1)EvFVxRjSzi=C>J@cMk87yJyz4~-Qcqlg}hXv}1CF`fEox?~SG{pae%Dy$pBG>tnWs3{>FohpTZSG
@fe-hAmws@4PFv7Mv`H@JnAXTbgKqwrl)IWaYE>+%OsO9KQH0000800Wn#Tn%mUY&Zb`0RI6102u%P0
B~t=FJEbHbY*gGVQep7UukY>bYEXCaCvo6!AitH486}+g!Qroon8cWL64rqlQ)qv+oo+`Ix{4xOTmvf
t*+w1T=EEymzS5G^8`)P&pI<U4bW6FD}<D?2Bn<DxqjsoD!!ql$SFiWD@8ezo0tdZE`Sp0iU&%=zJh5
Mg(E^6V#Ll+pBYVg*(RVFzPWxteKLr=uT17NQai#JO2H%$^t>kbFlCdRayLg5@77)Z?^5SnhsFz(_JI
dEKqS#uQGSCDc+Lsoerpw6J(yuVII!C7b}u8@K>~$Qkl)R)*@YZCXf1>s5u{}*Dxjlzn!*BNA;k4U#v
U0{YZf*+QtvkKXcD38Xbmz%=um^@b_s$AqiT^uT@R$=eDrOe>avtjThKL$%qaEE_1&#M8{GPTuke_Zm
y&Jz`<4^08S<AEM*DF38E)xR?h{Z;0|XQR000O81DB*+KNpR^IT!!{msJ1&8UO$QaA|NaUukZ1WpZv|
Y%gMUX>4R)Wo~vZaCzN4Yj5K?lHc<yxN0m~p0zcP4~K&>zC%CK)4|OqL3euh20@^eB|6r%ENS#e;>qQ
|Up+{Pq-dvSX77Nx0NsftvRJHl6^jmn;B}uDnK&))@}??xvg}0P%1mskM&xZ(Bwb#WBI}#HycKPhx1C
ty4EPuX!O6*{sdgfcH+|PPGLA*QtE;9HNm*7M&23Lk)U%?xy@hG|W7X=v>ZIEi`C5N$^G*(b?}t&HeC
Y0*q~;~lswiYi3+Od3noKq_dz+`-WK5K-Q$6milTETN<Xn7|Klidsr5;}58i0u&`D)#EO_FwR_|0FFG
P#wFNLpb(IW)iQi$Xo+A$g@=ee95;XBYgu+t+f~#}zCo*Gc;M3Z4${k^XwXPnW0hoA;-0{^RWYhq<_H
s;+{)&IL~3{qwXsucP=*mRZ$AH2_SI{?VOmT4bt%_BZ}GPSG!bt9gMuh5f?gxTsR%r~x~Ny)|W0L}`&
nE#O!JFg?GjWf?blx{cMFIX_aLd|>25Dr<|W;fa7B64JL4W@o^!{aM*|Nl~0OP1Ve4z}q~zEvvT6)62
Xzo~x06-%42!5aZoAa^}NDZ+AfDR(5d>3IPnB^9K<5;X|8l<xctl?PQl^Ntc*#GPz0mqKoCIcMQgO;D
gn7dC^9htQ*PLX`uR%%nnae3OF$S^uz0mkEe5yCN*iVR?ve01{pH9(tr}>jXaKV+pp`UN@Y7xGX=K9<
Kx3qJGralEN=$shtF?e!L+NI{aoN{cpHDt%M5MJ##WCdvDzoG2{25<B>8xQPYw+S=<-1kK!Rex#{r{D
e)^6c9s&TO->^JIWI*}<812jF5fsVj6+b%!2?&n=-~q|@VAlBK2ZJ340f1re5tnG4v;z@d!`~n96&Q#
Unwu60FzoaJkW>yL983&Q9xQyZdY2TfoH3d4N8)HL#!HdG&}*=wa{>R|oSdAXkimXe+8sn7LqF29Fpk
S)2L@*b!$3g<Xi3)BVX_niU`@YgV)1XH>+3=y*c<WR;|E@YQE4QIbSc6F=FJ31MI!#R5^Es=TYQRZ@b
==%MsD&4o<_?8rSdM0!&VlXIeDjX<@2hP<7RP0gacqd_{)CL0uD92Arg(R$WXxJ(Q1T20P7GMD+DvM;
jHtL6kw(1Ms3>Ub=TUXz!Kr87KSOpsnLU((x2IeX)2Qr_hpGLF$O^iX`irP-X?hYA0#+*fzAMU4*LY6
!S6&|?z=5mAo(EEzGIIH&jFU`K(*4oX+Q{z{Q{w*(LgC=!MfV!U<g_cHPoDhe9veR<{X2tRO58R(7@y
prr|QAMg-I_osmYJez#TMq*Ad6b{<{A@6dE<GuDYa8P0&6h&JO1aszRU0ubdb5;!!3{RQvjzUm={+9o
wPD4f$)I|*{aJ5)PoMTG6-nS{Omf=_^=5$G-KBk2s7DJbGx+ye=d5=lhwL6gwt75iuysbm~j6*7ji6I
1r9HrlkAOsh<)_u^|2#K>b}iEQFHs`tU5ZX}TV7A$IvF_<!Ttr7@HdVFI^1&^kPP3Yl_!>6_fw(WV>_
#XA#gvvg6-FIRG12NnYAWQ!+{AxJ++Hkqy>>o(lzD(|tyg+Fq03iU5fUH1L;C=^-f*6Rjk?0D=dT%`a
{=07!fj$s{THtN7ON#s#7zx`G)dpz^{Az9afn~Qzn{-HL*uh)`&RzmTBn$$gD*Cn_vVNl}GDXS2|9kP
>H=+VNeBTfpf>Q6reGVGelixJ~8bC~98niSi85zSnrHG%)>b^ux+~ug<TO`7qxYKS?xXYIv5}EWGq|;
pSehX2zixJ2SkW8#4C<Lf1Y$X%nApGM_XNTx^Z3)6$SdwsKiuq&Jk_nEGn}}RUCX}oPnP$M-1pYKweh
vsntD!t7cp@EA&z6=ToS_HZKy>b!CNe@8gK>bGOJ$%ye>sUlSv+p22KS@DhL&}o8%jS~!R3(4kZO*D*
q1D|ZyR5HHU5bK6MNR-5$p=XxuqwT06&YakH?OlP()EQ(DUP9yAM*<r6)%)9`b?ari!?flAwZ*zQIGb
!D}C&8;5L5J6IcEb2J9Y8fJVs(YAnOSH~E*S&+(+1im@(2CNPS(vqAhP<LHbg$@8*(%IfyMU*Rrr~NX
XU<U)I&p;-xZ=RXy;^+b74l!tKBp&7jh%3_16fSwTr<sjEZ*U+fkD0>UCk@62z?-wn_|3)n$Ey#o&d#
qyve{tzKIoUBH?0aJ@P}P+f(9LIQOScBBYHyZ2kqSX#g))IPs+?$aR8aO<o6-9=nQ{r?gYL3mk}5?dw
_`}jaaeKoQ>PU_{7-63jMjF06LT~uyGG&CT@tm0|4-tK8?`f)n-WJLuE`ZLuXvlR^`fmNR;|0J7S!}(
fb^<9kF26ieNgES#657@;h8&n%h%ma|a>t5h)ogA68(I+ghgiW-q48aHO_aOvDEW=fM+q_WIM=k8h3r
MdGT0IITs-vujnNJveR;prrI+gYn2>0m0uw$<yk~@Op`OefoGBzdL<(_36XuDsbloFOO017IYUZ=_>f
{U!v~<`yGS^y<FKgaP|jvT#lJMYS?C@CJbiUFxtb*q`9@+21GFy{U8d7jAY5C`y>mq-v^4i4oiZ?BEN
;eM=sQRnN3HO6$(-_eZkarp7TQrseF)7q8FTiE_C`dg{+idfw_jNH!g^n&d?X)RXe6}riDivENe_yo1
~bFynBh-1Yz8!FCb8Lpa9Y!;=3AiR0#|2DTJL1N%~HiNTO^yzE!EK17ufaJ<KN~wU(5eeyxCJXoYrF+
j00Cc?hFMPBwT2yJMOv<C9~~dtZT|lo047YY2`Zao+Z=GqD4C0xO_v%bWa`b8ygm%n|OAMg`}HEK5P=
;>9pr{urOfcmccy*JjYWoU-e^lRpBz3W(;v9}6~bPhWrf!FCuZ`4OeRSr{zt?hI!H0sM1-RUz^yNCP5
<4;#$fI#kuJfe;Ir69#1!uk$i#_5oyQRaxxyN5{MI+?Fdal-02wM>f#8v&d|Uo_y#Ky@?T?K2hul#9t
h!k#bpK5F^@)z|s3*@;r$CtIA8r<Qw$EdJ{m4$6#5aNeiB4p2;@ahA9KKsC`3{1&&;f)MQPu?IixZfV
3XvkIq?$1XK`eBv_X_NqUTi<|U$)gRCS42<~nNfz|?roo7zegELSrLMEX2YcRxE^}w_VI9(*L3|2xgA
8s&#lu1+M@Cxv2UCs(*DAf_?pdm#sQ8xVhL~)GOW@6lo8wR<24iyK7B|OaIMQ0|Rv2q;uP0<d~eKbBZ
*)cx#sDi@bI#}ee);RR~o12^2gbG}RrBO%#WB9kro8!BL6&5soL9-Sx3l=m0-k8eW%42Gac9G^d>I>5
F*<cL*it*X;jW2NAg2n+x`1{5y&yXZO?2kjDAppdjabx4SMQcCe*fl*kpDf15&!DM>E^sKUfxm8M9z0
VRR_)?cHr;W6IATQBnC1ry%7L-V^I&P93w@mJg^sRFHp;Xf<vovFXd^sw>V<d@CZUkiu=m%eSI2i~#L
uXplTg@^pD)VOUU1#Wiq>~Mo{LMWjiY63A@lFH-kmWB%snT&pNqEQS}z#3p=8uT<eW<Im=1`;#M2UNQ
VjzKEQEc@4!yE65^e4RcI=OD`h@a6Afs8a&ydUmD6pZAPro4xM}ku+|I1i>wn1;!NZ5#kd7cCfYolY}
dYIEj@j>I#VU^-uvea<k20Xs)i=;UMNTtDcJ<*Jqt??EKk!XSSBv`_e7<0w5bCt%P{cwKq;nc`xKf<8
k+u-TRXJh-isFDoqVcq1Ae09|`il~M!X!xS5MuRhf6u}lbya{K9XgSs;aNelJ%q-JX?hGD4-fx~(16A
xmZ4;2%JgRImLKpuO(5CqbNs;-Mj}wiu(1;Xf{8qn1xCze2it!_6u}{GaZG{?OQ_wjdpN5pU>uI>B;w
?yM+jO-Fa(Bf?A0?iGDIkT^O80I|27L9EB~kW(@p$#D{B<?q*LE#<ugeM$?~!SQBzY(awa5Z24-aKw0
zn}q26>=HE!K^(##MvLWO&G-+h3{;>yxTME5Z@q2nuEpGy;;BJOz`CD}X0^zV;ts&POm<#KEq^R9JJa
2CGB=Imy#K_8tgsM@YyLVB*P#FQvREIJ8^|2q4yDe#_#Alk|ZXCb?3AHYrkEsBRh@lb2x8ava4yMVOb
;S?7cXeOX{j1_FRld!QSh|Ll^_Kyq2OeEi`l+K0fG`)!_X)e$C6K4DpvVw*JR0W#cMD<v&PN4aZRRQI
#rP|`PQLSIzfS7cvWT^8dYVnj@gSQ;H6N6!d!2if(s@Pzdrvx);;TUTw~?U#b<x!)YDNgZz=fw-zBes
bG}vE(JcX-+JHMe#r3aKV{l&<aKypPX?8X8E|+6Nif2&tUcPglM32y%Ma-Vkmiu+oD=y5hvbJ!(FzzY
9PsS)>3pqxE`9=29%K)8XV)n$gFG;!ykwS2F<CIK4DS61C6r_K}_D&#*scokt~3E7g4qos|LlYiL`<6
APJOO4S#vca+YkbtWWm9K>%07U(;HZO2!+^mG3Yc2aA$-Yz)Eq7O3Ws^UEx!BBoW<wTaO-Dd8SWDq`;
kOEAzgB-^gaS$IgZHtf~R28>o5Lk5CwSED_wB~h_QP$5!@J=m{yWt`9HAMhx*4KO8MC$NQTvzgZQF5X
@&@sxYZl}&1p9qMr$j&OI&U@6s+I_))g<GT#ArAD>ha_xg?|B~P(