import { useMemo } from "react";
import Content from "@components/Docs/Content";
import { CodeBlock, codepen, monokaiSublime } from "react-code-blocks";
import { Flex, Text, Box } from "@chakra-ui/react";
import { useTable } from "react-table";
import styled from "styled-components";

const APIDocs = () => {
  const columns = useMemo(
    () => [
      {
        key: 1,
        Header: "Param",
        accessor: "col1",
      },
      {
        key: 2,
        Header: "Type",
        accessor: "col2",
      },
      {
        key: 3,
        Header: "Required",
        accessor: "col3",
      },
      {
        key: 4,
        Header: "Description",
        accessor: "col4",
      },
    ],
    []
  );

  const transcribeData = useMemo(
    () => [
      {
        key: 1,
        col1: "dataset_name",
        col2: "string",
        col3: "Yes",
        col4: "Name of the dataset that should be used (LibriSpeech)",
      },
      {
        key: 2,
        col1: "model_name",
        col2: "string",
        col3: "Yes",
        col4: "Name of the model that should be used (quartznet, jasper)",
      },
      {
        key: 3,
        col1: "file_datatype",
        col2: "string",
        col3: "Yes",
        col4: "Type of file being sent to be processed - uploaded Audio file, uploaded Video file, or Youtube URL (a_upload, v_upload, y_video)",
      },
      {
        key: 4,
        col1: "file",
        col2: "string/blob",
        col3: "Yes",
        col4: "The Blob content of uploaded file or URL of Youtube video",
      },
    ],
    []
  );

  const exportData = useMemo(
    () => [
      {
        key: 1,
        col1: "text",
        col2: "string",
        col3: "Yes",
        col4: "Transcription text to be exported to pdf format.",
      },
    ],
    []
  );

  return (
      <div>
        <Content Title="Transcribe File">
          <Flex flexDirection="column">
            <Flex flexDirection="row">
              <Text as="b" color="yellow.700">
                POST
              </Text>
              <Box ml="1rem" bg="gray.100">
                <Text color="gray.500">/transcribe_file</Text>
              </Box>
            </Flex>
            <Text as="b" mt="1rem">
              Parameters
            </Text>
            <Styles>
              <Table columns={columns} data={transcribeData} />
            </Styles>
            <Text as="b" mt="1.5rem">
              Example Request
            </Text>
            <CodeBlock
              text={` curl --location --request POST 'http://subme-api.herokuapp.com/transcribe_file'  \u005C
    --header 'Accept: application/json'  \u005C
    --header 'Content-Type: application/json'  \u005C
    --data-raw '{
      "dataset_name": "LibriSpeech",
      "model_name": "quartznet",
      "file_datatype": "y_video",
      "file": "http://www.youtube.com/watch?v=fk2764Rjdas"
  }'`}
              language={"javascript"}
              showLineNumbers={false}
              theme={codepen}
              codeBlock
            />
            <Text as="b" mt="1.5rem">
              Example Response
            </Text>
            <CodeBlock
              text={` {
            latency: 6.39613534,
            transcript: "Transcription text from the API response"
}`}
              language={"json"}
              showLineNumbers={false}
              theme={monokaiSublime}
              codeBlock
            />
          </Flex>
        </Content>

        <Content Title="Export File">
          <Flex flexDirection="column">
            <Flex flexDirection="row">
              <Text as="b" color="yellow.700">
                POST
              </Text>
              <Box ml="1rem" bg="gray.100">
                <Text color="gray.500">/export_file</Text>
              </Box>
            </Flex>
            <Text as="b" mt="1rem">
              Parameters
            </Text>
            <Styles>
              <Table columns={columns} data={exportData} />
            </Styles>
            <Text as="b" mt="1.5rem">
              Example Request
            </Text>
            <CodeBlock
              text={` curl --location --request POST 'http://subme-api.herokuapp.com/export_file'  \u005C
    --header 'Accept: application/json'  \u005C
    --header 'Content-Type: application/json'  \u005C
    --data-raw '{
      "text": "Sample text"
}'`}
              language={"javascript"}
              showLineNumbers={false}
              theme={codepen}
              codeBlock
            />
            <Text as="b" mt="1.5rem">
              Example Response
            </Text>
            <CodeBlock
              text={` {
            fileLink: "https://storage.googleapis.com/subme-transcription/259571208.pdf"
}`}
              language={"javascript"}
              showLineNumbers={false}
              theme={monokaiSublime}
              codeBlock
            />
          </Flex>
        </Content>
      </div>
  );
};

interface TableProps {
  columns: {
    key: number | string;
    Header: string;
    accessor: string;
  }[];
  data: {
    key: number;
    col1: string;
    col2: string;
    col3: string;
    col4: string;
  }[];
}

function Table({ columns, data }: TableProps) {
  const { getTableProps, getTableBodyProps, headerGroups, rows, prepareRow } =
    useTable({
      columns,
      data,
    });
  return (
    <table {...getTableProps()}>
      <thead>
        {headerGroups.map((headerGroup) => (
          <tr
            key={headerGroup.getHeaderGroupProps()}
            {...headerGroup.getHeaderGroupProps()}
          >
            {headerGroup.headers.map((column) => (
              <th key={column.getHeaderProps()} {...column.getHeaderProps()}>
                {column.render("Header")}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map((row) => {
          prepareRow(row);
          return (
            <tr key={row.getRowProps()} {...row.getRowProps()}>
              {row.cells.map((cell) => {
                return (
                  <td key={cell.getCellProps()} {...cell.getCellProps()}>
                    {cell.render("Cell")}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

const Styles = styled.div`
  padding: 1rem;

  table {
    border-spacing: 0;
    border: 1px solid black;

    tr {
      :last-child {
        td {
          border-bottom: 0;
        }
      }
    }

    th,
    td {
      margin: 0;
      padding: 0.5rem;
      border-bottom: 1px solid black;
      border-right: 1px solid black;

      :last-child {
        border-right: 0;
      }
    }
  }
`;

export default APIDocs;
