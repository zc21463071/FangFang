#ifndef	__UTILITYFUNCTION_H__
#define __UTILITYFUNCTION_H__

#include "FPAType.h"
#include "FPAGlobal.h"

int IMGW;

int IMGH;

BYTE *g_lpOrgFinger;

void Setup(int W, int H, BYTE *lpData)
{
	IMGW = W;
	IMGH = H;
	g_lpOrgFinger = lpData;
}
//
//	IsFork: 判断某点是否为叉点
//
bool   IsFork(BYTE *lpNow)
{
/////////////////////////////////////////////////////////////////////////////
//	lpNow : [in] 当前点的地址
/////////////////////////////////////////////////////////////////////////////
	int i, sum;
	// 某点周围8个点的地址偏移
	int	SiteD8[8] = {IMGW-1, IMGW, IMGW+1, 1, -IMGW+1, -IMGW, -IMGW-1, -1};
	
	// 8个点所有相邻两个点的差的绝对值的和如果为6*255则为叉点
	sum = 0;
	for(i = 0; i < 8; i++)
	{
		sum += abs(*(lpNow + SiteD8[(i+1)%8]) - *(lpNow + SiteD8[i]));
	}
	if(sum == 255*6)
	{
		return true;
	}
	else
	{
		return false;
	}
}

//
//	IsEnd: 判断某点是否为端点
//
bool   IsEnd(BYTE *lpNow)
{
/////////////////////////////////////////////////////////////////////////////
//	lpNow : [in] 当前点的地址
/////////////////////////////////////////////////////////////////////////////
	int i, sum;
	// 某点周围8个点的地址偏移
	int	SiteD8[8] = {IMGW-1, IMGW, IMGW+1, 1, -IMGW+1, -IMGW, -IMGW-1, -1};

	// 8个点所有相邻两个点的差的绝对值的和如果为2*255则为端点
	sum = 0;
	for(i = 0; i < 8; i++)
	{
		sum += abs(*(lpNow + SiteD8[(i+1)%8]) - *(lpNow + SiteD8[i]));
	}
	if(sum == 255*2)
	{
		return true;
	}
	else
	{
		return false;
	}
}

//
//	GetNext: 在纹线上根据当前点和前驱点地址找到下一个点的地址
//
int  GetNext(BYTE *lpNow, BYTE *lpLast, BYTE **lppNext)
{
/////////////////////////////////////////////////////////////////////////////
//	lpNow : [in] 当前点的地址
//	lpLast : [in] 前驱点的地址
//	lppNext : [out] 下一个点的地址的指针
/////////////////////////////////////////////////////////////////////////////
	int  i;
	BYTE   *lpSrc = NULL;
	int  n;
	BYTE   *Temp[8];	// 某点周围8个点的地址
	int	SiteD8[8] = {IMGW-1, IMGW, IMGW+1, 1, -IMGW+1, -IMGW, -IMGW-1, -1};

	*lppNext = NULL;
	
	// 找到当前点周围的是黑点并且不是前驱点的黑点，保存到数组
	n = 0;
	for(i = 0; i < 8; i++)
	{
		lpSrc = lpNow + SiteD8[i];
		if(*lpSrc == 0 && lpSrc != lpLast)
		{
			Temp[n] = lpSrc;
			n++;
		}
	}
	
	if(n == 0)	// 没有找到其他黑点则表示没有后继点
	{
		*lppNext = NULL;
		return 1;
	}
	else if(n == 1) // 找到了一个，则其为后继点
	{
		*lppNext = Temp[0];
		return 0;
	}
	else // 找到多个点则返回错误
	{
		*lppNext = NULL;
		return 1;
	}
	
	return 0;
}

//
//	GetByDis: 在纹线上找到距离当前端点为d的点的地址
//
int  GetByDis(BYTE *lpEnd, BYTE **lppPos, int d)
{
/////////////////////////////////////////////////////////////////////////////
//	lpEnd : [in] 当前端点的地址
//	lppPos : [out] 目的点的地址(如果在距离d内有端点或者叉点，则返回他们的邻点)
//	d : [in] 距离（点数）
/////////////////////////////////////////////////////////////////////////////
	int  i, n;
	BYTE   *lpNow = NULL;
	BYTE   *lpLast = NULL;
	BYTE   *lpNext = NULL;
	int	SiteD8[8] = {IMGW-1, IMGW, IMGW+1, 1, -IMGW+1, -IMGW, -IMGW-1, -1};

	*lppPos = NULL;
	
	// 找到当前点周围的是黑点并且不是前驱点的黑点，保存到数组
	n = 0;
	for(i = 0; i < 8; i++)
	{
		if(*(lpEnd + SiteD8[i]) == 0)
		{
			lpNow = lpEnd + SiteD8[i];
			n++;
		}
	}
	if(n != 1) // 数目如果不是1则返回错误
	{
		return 1;
	}
	// 找到的黑点地址为lpNow
	// 端点作为前驱点,地址保存到lpLast
	lpLast = lpEnd; 
	// 循环找d次后继点
	for(i = 0; i < d; i++)
	{
		// 如果是叉点则跳出循环
		if(IsFork(lpNow))
		{
			break;
		}
		// 成功找到下一个点则更改当前点和前驱点
		if(GetNext(lpNow, lpLast, &lpNext) == 0)
		{
			lpLast = lpNow;
			lpNow = lpNext;
		}
		else
		{
			break;
		}

	}
	// 如果找到的点为叉点，则返回其前驱点
	if(IsFork(lpNow))
	{
		*lppPos = lpLast;
	}	
	else	// 否则返回当前点
	{
		*lppPos = lpNow;
	}

	return 0;
}

//
//	PX: 根据地址得到当前点的横坐标
//
int  PX(BYTE *lpPos)
{
/////////////////////////////////////////////////////////////////////////
//	lpPos: [in] 当前点地址
/////////////////////////////////////////////////////////////////////////
	return ((lpPos - g_lpOrgFinger) % IMGW);
}

//
//	PY: 根据地址得到当前点的纵坐标
//
int  PY(BYTE *lpPos)
{
/////////////////////////////////////////////////////////////////////////
//	lpPos: [in] 当前点地址
/////////////////////////////////////////////////////////////////////////
	return ((lpPos - g_lpOrgFinger) / IMGW);
}

//
//	GetJiajiao: 求两个角度的夹角(0 - 90)
//
int  GetJiajiao(int angle1, int angle2)
{
/////////////////////////////////////////////////////////////////////////
//	angle1: [in] 角度一 （0 - 180）
//	angle2: [in] 角度二 （0 - 180）
/////////////////////////////////////////////////////////////////////////
	int  a;

	a = abs(angle1-angle2);
	if(a > 90)
	{
		a = 180-a;
	}
	return a;
}

//
//	AngleAbs360: 求两个角度的夹角(0 - 180)
//
int  AngleAbs360(int angle1, int angle2)
{
/////////////////////////////////////////////////////////////////////////
//	angle1: [in] 角度一 （0 - 360）
//	angle2: [in] 角度二 （0 - 360）
/////////////////////////////////////////////////////////////////////////
	int a;
	a = abs(angle1 - angle2);
	if(a > 180)
	{
		return (360 - a);
	}
	else
	{
		return a;
	}
}

//
//	GetAngleDis： 两个角度间的距离(0-360)  (逆时针方向)
//
int  GetAngleDis(int angleBegin, int angleEnd)
{
/////////////////////////////////////////////////////////////////////////
//	angleBegin: [in] 起始角度 （0 - 360）
//	angleEnd: [in] 终结角度 （0 - 360）
/////////////////////////////////////////////////////////////////////////
	int  a;

	a = angleEnd-angleBegin;
	if(a < 0)
	{
		a += 360;
	}
	return a;
}

//
//	GetAngle： 两个点连线与x轴方向的角度
//
int  GetAngle(int x0, int y0, int x1, int y1)
{
/////////////////////////////////////////////////////////////////////////
//	x0: [in] 第一个点横坐标
//	y0: [in] 第一个点纵坐标
//	x1: [in] 第二个点横坐标
//	y1: [in] 第二个点纵坐标
/////////////////////////////////////////////////////////////////////////
	double angle;

	angle = atan2((double)(y1-y0), (double)(x1-x0));
	// 弧度转化到0 - 2*PI
	if(angle < 0)
	{
		angle += 2*PI;
	}
	// 弧度转化为角度
	return (int)(angle*EPI + 0.5);
}

//
//	DIndex: 对角度进行量化，得到量化的索引号(0-12)
//
int  DIndex(int angle)
{
/////////////////////////////////////////////////////////////////////////
//	angle: [in] 角度 （0 - 180）
/////////////////////////////////////////////////////////////////////////
	if(angle >= 173 || angle < 8)
	{
		return 0;
	}
	else
	{
		return ((angle-8)/15 + 1);
	}
}

#endif