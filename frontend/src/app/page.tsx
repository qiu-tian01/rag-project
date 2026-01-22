'use client';

import React, { useState } from 'react';
import { Input, Button, Card, Typography, Space, Dropdown, message, Tag } from 'antd';
import type { MenuProps } from 'antd';
import { Send, Search, Zap, Cpu, ChevronDown, FileText } from 'lucide-react';

const { Title, Text } = Typography;
const { TextArea } = Input;

// API基础URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Source {
  chunk_id: string;
  document_name: string;
  section_path: string[];
  text: string;
  similarity: number;
  page_num?: number;
}

interface ChatResponse {
  answer: string;
  thoughts?: string;
  citations?: number[];
  sources: Source[];
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchMode, setSearchMode] = useState<number>(2); // 1=纯向量搜索, 2=混合检索+rerank
  const [llmModel, setLlmModel] = useState<number>(2); // 1=qwen-max, 2=qwen-plus, 3=qwen-turbo
  const [productName, setProductName] = useState('');
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const searchModeItems: MenuProps['items'] = [
    { 
      key: '1', 
      label: '纯向量搜索',
      icon: <Search size={14} className="text-blue-500" />
    },
    { 
      key: '2', 
      label: '混合检索+rerank',
      icon: <Zap size={14} className="text-orange-500" />
    },
  ];

  const llmModelItems: MenuProps['items'] = [
    { 
      key: '1', 
      label: 'qwen-max', 
      icon: <Cpu size={14} className="text-purple-500" /> 
    },
    { 
      key: '2', 
      label: 'qwen-plus', 
      icon: <Zap size={14} className="text-orange-500" /> 
    },
    { 
      key: '3', 
      label: 'qwen-turbo', 
      icon: <Zap size={14} className="text-yellow-500" /> 
    },
  ];

  const handleSearch = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          search_mode: searchMode,
          llm_model: llmModel,
          product_name: productName.trim() || undefined,
          history: [],
          stream: false,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: '请求失败' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setResponse(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '搜索失败，请稍后重试';
      setError(errorMessage);
      message.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getSearchModeLabel = (mode: number) => {
    return mode === 1 ? '纯向量搜索' : '混合检索+rerank';
  };

  const getLlmModelLabel = (model: number) => {
    const labels: Record<number, string> = {
      1: 'qwen-max',
      2: 'qwen-plus',
      3: 'qwen-turbo',
    };
    return labels[model] || 'qwen-plus';
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-[#141414] text-white">
      <div className="w-full max-w-4xl flex flex-col items-center gap-8">
        <div className="text-center space-y-2 mb-8">
          <Title level={1} style={{ color: 'white', marginBottom: 0, fontSize: '3rem' }}>RAG 智能问答</Title>
          <Title level={2} style={{ color: '#888', marginTop: 0, fontWeight: 400 }}>基于文档检索的增强问答系统</Title>
        </div>

        <div className="w-full relative">
          <div className="bg-[#1f1f1f] rounded-2xl p-4 border border-[#333] focus-within:border-blue-500 transition-colors shadow-lg">
            <TextArea 
              placeholder="请输入您的问题，例如：招商小程序的功能有哪些？" 
              autoSize={{ minRows: 2, maxRows: 6 }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              bordered={false}
              className="text-lg !text-white placeholder:!text-gray-500 !bg-transparent !resize-none mb-4"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSearch();
                }
              }}
            />
            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-center">
                <Space wrap>
                  <Dropdown 
                    menu={{ 
                      items: searchModeItems,
                      selectedKeys: [String(searchMode)],
                      onClick: ({ key }) => {
                        setSearchMode(Number(key));
                      }
                    }} 
                    trigger={['click']}
                    getPopupContainer={(triggerNode) => triggerNode.parentElement || document.body}
                  >
                    <a onClick={(e) => e.preventDefault()} className="bg-[#2a2a2a] rounded-full px-3 py-1 text-sm text-gray-400 flex items-center gap-2 cursor-pointer hover:bg-[#333] transition-colors select-none">
                      <Search size={14} />
                      {getSearchModeLabel(searchMode)}
                      <ChevronDown size={12} className="opacity-50" />
                    </a>
                  </Dropdown>

                  <Dropdown 
                    menu={{ 
                      items: llmModelItems,
                      selectedKeys: [String(llmModel)],
                      onClick: ({ key }) => {
                        setLlmModel(Number(key));
                      }
                    }} 
                    trigger={['click']}
                    getPopupContainer={(triggerNode) => triggerNode.parentElement || document.body}
                  >
                    <a onClick={(e) => e.preventDefault()} className="bg-[#2a2a2a] rounded-full px-3 py-1 text-sm text-gray-400 flex items-center gap-2 cursor-pointer hover:bg-[#333] transition-colors select-none">
                      {llmModel === 1 ? <Cpu size={14} /> : <Zap size={14} />}
                      {getLlmModelLabel(llmModel)}
                      <ChevronDown size={12} className="opacity-50" />
                    </a>
                  </Dropdown>
                </Space>
                <Button 
                  type="text"  
                  shape="circle"
                  icon={<Send size={20} />} 
                  onClick={handleSearch}
                  loading={loading}
                  className="!text-white hover:!bg-[#333] !flex !items-center !justify-center"
                  size="large"
                />
              </div>
              <Input
                placeholder="产品名称（可选，用于过滤相关文档）"
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
                bordered={false}
                className="!bg-[#2a2a2a] !text-white placeholder:!text-gray-500 rounded-lg px-3 py-1 text-sm"
                prefix={<FileText size={14} className="text-gray-400" />}
              />
            </div>
          </div>
        </div>

        <Text className="text-gray-600 text-xs mt-4">Powered by Qwen LLM and RAG Pipeline</Text>

        {/* Error Display */}
        {error && (
          <Card className="w-full !bg-red-900/20 !border-red-500">
            <Text className="!text-red-400">错误: {error}</Text>
          </Card>
        )}

        {/* Results Section */}
        {response && (
          <div className="w-full mt-8 space-y-4 animate-in fade-in slide-in-from-bottom-4">
            <div className="flex items-center justify-between">
              <Title level={4} style={{ color: 'white', margin: 0 }}>回答结果</Title>
              <Button type="link" onClick={() => setResponse(null)} className="!text-gray-400">清除</Button>
            </div>

            {/* Answer */}
            <Card 
              size="small" 
              className="!bg-[#1f1f1f] !border-[#333] hover:!border-[#444] transition-colors" 
              styles={{ body: { color: '#ddd' } }}
            >
              <div className="mb-3">
                <Text strong className="!text-blue-400 text-base">答案：</Text>
              </div>
              <Text className="!text-gray-200 text-base leading-relaxed whitespace-pre-wrap">
                {response.answer}
              </Text>
            </Card>

            {/* Thoughts */}
            {response.thoughts && (
              <Card 
                size="small" 
                className="!bg-[#1f1f1f] !border-[#333] hover:!border-[#444] transition-colors" 
                styles={{ body: { color: '#ddd' } }}
              >
                <div className="mb-3">
                  <Text strong className="!text-purple-400 text-base">推理过程：</Text>
                </div>
                <Text className="!text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
                  {response.thoughts}
                </Text>
              </Card>
            )}

            {/* Citations */}
            {response.citations && response.citations.length > 0 && (
              <div className="flex items-center gap-2">
                <Text strong className="!text-gray-400">引用页码：</Text>
                <Space wrap>
                  {response.citations.map((page, idx) => (
                    <Tag key={idx} color="blue">第 {page} 页</Tag>
                  ))}
                </Space>
              </div>
            )}

            {/* Sources */}
            {response.sources && response.sources.length > 0 && (
              <div className="mt-4">
                <Title level={5} style={{ color: 'white', marginBottom: 12 }}>来源文档 ({response.sources.length})</Title>
                <Space direction="vertical" className="w-full" size="small">
                  {response.sources.map((source, idx) => (
                    <Card 
                      key={source.chunk_id || idx}
                      size="small" 
                      className="!bg-[#1f1f1f] !border-[#333] hover:!border-[#444] transition-colors" 
                      styles={{ body: { color: '#ddd', padding: '12px' } }}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1">
                          <Text strong className="!text-blue-400 text-sm">{source.document_name}</Text>
                          {source.page_num && (
                            <Tag color="cyan" className="ml-2">第 {source.page_num} 页</Tag>
                          )}
                          {source.similarity && (
                            <Tag color="green" className="ml-2">相似度: {(source.similarity * 100).toFixed(1)}%</Tag>
                          )}
                        </div>
                      </div>
                      {source.section_path && source.section_path.length > 0 && (
                        <div className="mb-2">
                          <Text className="!text-gray-500 text-xs">
                            章节: {source.section_path.join(' > ')}
                          </Text>
                        </div>
                      )}
                      <Text className="!text-gray-300 text-sm leading-relaxed line-clamp-3">
                        {source.text}
                      </Text>
                    </Card>
                  ))}
                </Space>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
