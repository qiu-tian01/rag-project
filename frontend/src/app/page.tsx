'use client';

import React, { useState } from 'react';
import { Input, Button, Card, Typography, Space, Dropdown } from 'antd';
import type { MenuProps } from 'antd';
import { Send, Brain, Zap, Cpu, ChevronDown } from 'lucide-react';

const { Title, Text } = Typography;
const { TextArea } = Input;

interface SearchResult {
  id: number;
  title: string;
  content: string;
  score: number;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [effort, setEffort] = useState('Medium');
  const [model, setModel] = useState('2.5 Flash');

  const effortItems: MenuProps['items'] = [
    { key: 'Low', label: 'Low' },
    { key: 'Medium', label: 'Medium' },
    { key: 'High', label: 'High' },
  ];

  const modelItems: MenuProps['items'] = [
    { 
      key: '2.0 Flash', 
      label: '2.0 Flash', 
      icon: <Zap size={14} className="text-yellow-500" /> 
    },
    { 
      key: '2.5 Flash', 
      label: '2.5 Flash', 
      icon: <Zap size={14} className="text-orange-500" /> 
    },
    { 
      key: '2.5 Pro', 
      label: '2.5 Pro', 
      icon: <Cpu size={14} className="text-purple-500" /> 
    },
  ];

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    // Mock search for now
    setTimeout(() => {
      setResults([
        { id: 1, title: 'Document 1', content: 'This is the content of document 1 matching your query.', score: 0.95 },
        { id: 2, title: 'Document 2', content: 'Another relevant document found in the knowledge base.', score: 0.88 },
        { id: 3, title: 'Euro 2024 Report', content: 'Spain won Euro 2024. Dani Olmo was one of the top scorers.', score: 0.99 },
      ]);
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-[#141414] text-white">
      <div className="w-full max-w-3xl flex flex-col items-center gap-8">
        <div className="text-center space-y-2 mb-8">
          <Title level={1} style={{ color: 'white', marginBottom: 0, fontSize: '3rem' }}>Welcome.</Title>
          <Title level={2} style={{ color: '#888', marginTop: 0, fontWeight: 400 }}>How can I help you today?</Title>
        </div>

        <div className="w-full relative">
          <div className="bg-[#1f1f1f] rounded-2xl p-4 border border-[#333] focus-within:border-blue-500 transition-colors shadow-lg">
            <TextArea 
              placeholder="Who won the Euro 2024 and scored the most goals?" 
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
            <div className="flex justify-between items-center">
                <Space>
                   <Dropdown 
                      menu={{ 
                        items: effortItems, 
                        selectable: true, 
                        defaultSelectedKeys: ['Medium'],
                        onClick: (e) => setEffort(e.key)
                      }} 
                      trigger={['click']}
                   >
                     <div className="bg-[#2a2a2a] rounded-full px-3 py-1 text-sm text-gray-400 flex items-center gap-2 cursor-pointer hover:bg-[#333] transition-colors select-none">
                        <Brain size={14} />
                        Effort {effort}
                        <ChevronDown size={12} className="opacity-50" />
                     </div>
                   </Dropdown>

                   <Dropdown 
                      menu={{ 
                        items: modelItems, 
                        selectable: true, 
                        defaultSelectedKeys: ['2.5 Flash'],
                        onClick: (e) => setModel(e.key)
                      }} 
                      trigger={['click']}
                   >
                     <div className="bg-[#2a2a2a] rounded-full px-3 py-1 text-sm text-gray-400 flex items-center gap-2 cursor-pointer hover:bg-[#333] transition-colors select-none">
                        {model.includes('Pro') ? <Cpu size={14} /> : <Zap size={14} />}
                        Model {model}
                        <ChevronDown size={12} className="opacity-50" />
                     </div>
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
          </div>
        </div>

        <Text className="text-gray-600 text-xs mt-4">Powered by Google Gemini and LangChain LangGraph.</Text>

        {/* Results Section */}
        {results.length > 0 && (
          <div className="w-full mt-8 space-y-4 animate-in fade-in slide-in-from-bottom-4">
             <div className="flex items-center justify-between">
                <Title level={4} style={{ color: 'white', margin: 0 }}>Search Results</Title>
                <Button type="link" onClick={() => setResults([])}>Clear</Button>
             </div>
             {results.map((res) => (
               <Card 
                key={res.id} 
                size="small" 
                className="!bg-[#1f1f1f] !border-[#333] hover:!border-[#444] transition-colors" 
                styles={{ body: { color: '#ddd' } }}
               >
                 <div className="flex justify-between items-start mb-2">
                   <Text strong className="!text-blue-400 text-lg">{res.title}</Text>
                   <div className="bg-[#2a2a2a] px-2 py-0.5 rounded text-xs text-gray-400">
                      Score: {res.score}
                   </div>
                 </div>
                 <Text className="!text-gray-300">{res.content}</Text>
               </Card>
             ))}
          </div>
        )}
      </div>
    </div>
  );
}
