# tools.py
from typing import Dict, Any
import datetime
from utils import get_ai_result, ai_action, ai_row_submit

def register_dev_tools(mcp):
  @mcp.tool()
  async def get_dev_list(
    username: str,
    dept: str,
    devdes: str = ""
  ) -> Dict[str, Any]:
    """
    获取车间的设备列表

    Args:
      username: 用户名，必填，用于权限验证。
      dept: 车间编码，必填。例如："AC" 表示 AC 车间。
      devdes: 设备名称，用于筛选特定设备。如果未提供，则返回所有设备。默认为空字符串。

    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP"
    dmNum = 110
    para = [dept, devdes]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_dev_mnt_remain(
    username: str,
    dept: str,
    devdes: str = ""
  ) -> Dict[str, Any]:
    """
    已经创建，等待处理，尚未完成处理的记录
    Args:
      username: 用户名，必填，用于权限验证。
      dept: 车间编码，必填。例如："AC" 表示 AC 车间。
      devdes: 设备名称，用于筛选特定设备。如果未提供，则返回所有设备。默认为空字符串。
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP"
    dmNum = 120
    para = [dept, devdes]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_dev_mnt_list(
    username: str,
    dept: str,
    startdate: str,
    enddate: str,
    devdes: str = ""
  ) -> Dict[str, Any]:
    """
    按照车间、设备名称、日期范围获取设备维保记录
    Args:
      username: 用户名，必填，用于权限验证。
      dept: 车间编码，必填。例如："AC" 表示 AC 车间。
      startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
      enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
      devdes: 设备名称，用于筛选特定设备。如果未提供，则返回所有设备。默认为空字符串。
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP"
    dmNum = 130
    para = [dept, devdes, startdate, enddate]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def dev_mnt_reg(
    username: str,
    dept: str,
    devno: str,
    mnttyp: str,
    mntdes: str,
    plnres: str,
    plntim: datetime.datetime
  ) -> Dict[str, Any]:
    """
    等待处理的设备维保记录，已经创建维保记录，尚未完成处理的记录
    Args:
      username: 用户名，必填，用于权限验证。
      dept: 车间编码，必填。例如："AC" 表示 AC 车间。
      devno: 设备编码，必填，要确保设备编码存在，根据get_dev_list获取当前车间的设备列表检查，明确取已经存在的设备编码
      mnttyp: 维保类型，必填
      mntdes: 维保描述，必填
      plnres: 计划安排的人员
      plntim: 计划维保时间，如果未明确参数值，则使用当前时间
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False
    """
    dmCode = "LINKAIMCP"
    dmNum = 110
    action = 501
    # 参数与模型对应，前面两个参数是主模型使用的，501的动作不用
    para = ["", "", mnttyp, mntdes, plntim, plnres]
    rowdata = {"车间": dept, "设备编号": devno}
    return await ai_action(dmCode, dmNum, action, para, rowdata, username)

  @mcp.tool()
  async def dev_mnt_complete(
    username: str,
    docnum: str,
    cplres: str,
    remark: str
  ) -> Dict[str, Any]:
    """
    根据维保登记的单据号，进行维保完成登记，记录完成人、完成时间、备注
    Args:
      username: 用户名，必填，用于权限验证。
      docnum: 维保单据编号，以 MFH 开头，必填
      cplres: 完成人，必填
      remark: 备注，如果未明确参数值，则使用空字符串
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False
    """
    dmCode = "LINKAIMCP"
    dmNum = 120
    scriptType = 2  # 2 模型的记录编辑脚本
    # 参数与模型对应，改功能不使用参数
    para = ["", ""]
    # 通过行记录的模式传递数据
    rowdata = {"DOCNUM": docnum, "CPLRES": cplres, "REMARK": remark}
    return await ai_row_submit(dmCode, dmNum, scriptType, para, rowdata, username)