# tools.py
from typing import Dict, Any
import datetime
from utils import get_ai_result, get_ai_action
from mcp.server.fastmcp import FastMCP

def register_data_ana_tools(mcp: FastMCP):
  @mcp.tool()
  async def get_sales_data(
    startdate: str,
    enddate: str,
    company: str = "",
    cus: str = "",
    item: str = "",
    username: str = "DINA"
  ) -> Dict[str, Any]:
    """
    获取销售数据，提问包含具体的公司、客户、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
      startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
      enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
      company: 公司编码或公司名称，如果未明确参数值，则使用空字符串
      cus: 客户编码或客户名称，如果未明确参数值，则使用空字符串
      item: 产品编码或产品名称，如果未明确参数值，则使用空字符串
      username: 用户名，默认为 "DINA"
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP10X.SALES"
    dmNum = 10
    para = [startdate, enddate, company, cus, item]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_pur_data(
    startdate: str,
    enddate: str,
    company: str = "",
    sup: str = "",
    item: str = "",
    username: str = "DINA"
  ) -> Dict[str, Any]:
    """
    获取采购数据，提问包含具体的公司、供应商、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
      startdate: 起始日期，格式为 "YYYY-MM-DD"
      enddate: 结束日期，格式为 "YYYY-MM-DD"
      company: 公司编码或公司名称，如果未明确参数值，则使用空字符串
      sup: 供应商编码或供应商名称，如果未明确参数值，则使用空字符串
      item: 产品编码或产品名称，如果未明确参数值，则使用空字符串
      username: 用户名，默认为 "DINA"
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP10X.PUR"
    dmNum = 10
    para = [startdate, enddate, company, sup, item]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_inv_data(
    site: str = "",
    loc: str = "",
    item: str = "",
    username: str = "DINA"
  ) -> Dict[str, Any]:
    """
    获取库存数据，提问包含具体的地点、仓库、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
      site: 地点编码或地点名称，如果未明确参数值，则使用空字符串
      loc: 仓库，如果未明确参数值，则使用空字符串
      item: 产品编码、产品类别或产品名称，如果未明确参数值，则使用空字符串
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP10X.INV"
    dmNum = 10
    para = [site, loc, item]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_inv_detail(
    site: str,
    warehouse: str,
    username: str = "DINA"
  ) -> Dict[str, Any]:
    """
    按地点、仓库获取产品库存明细数据
    如果没有明确的warehouse参数，则从get_inv_data的返回结果中查找warehouse参数的具体值
    Args:
      site: 地点编码，必填
      warehouse: 仓库编码，必填
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP10X.INV"
    dmNum = 20
    para = [site, warehouse]
    return await get_ai_result(dmCode, dmNum, para, username)

  @mcp.tool()
  async def get_item_trx(
    site: str,
    item: str,
    startdate: str,
    enddate: str,
    username: str = "DINA"
  ) -> Dict[str, Any]:
    """
    按地点、产品编码和时间范围获取产品库存交易变动记录
    如果没有明确的产品编码参数，则从get_inv_detail的返回结果中查找产品编码参数的具体值
    Args:
      site: 地点，必填
      item: 产品编码，必填
      startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
      enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    dmCode = "LINKAIMCP10X.INV"
    dmNum = 30
    para = [site, item, startdate, enddate]
    return await get_ai_result(dmCode, dmNum, para, username)
